// Mini-Vinyl — NFC-проигрыватель на ESP32.
// Пластинка с NFC-меткой → трек или папка MP3 с карты памяти. Свет «вращается» вокруг пластинки.
// Схема подключения — в docs/wiring.md, формат карты — в firmware/sdcard/tags.csv.
#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <Preferences.h>
#include <vector>
#include <algorithm>
#include <Adafruit_PN532.h>
#include <FastLED.h>
#include <AudioFileSourceSD.h>
#include <AudioFileSourceID3.h>
#include <AudioGeneratorMP3.h>
#include <AudioOutputI2S.h>

// ---------------------------------------------------------------- выводы
const int PIN_SDA = 21, PIN_SCL = 22, PIN_NFC_IRQ = 32, PIN_NFC_RST = 33;   // PN532 (I2C)
const int PIN_SD_CS = 5;                                                    // microSD (SPI: 18, 23, 19)
const int PIN_I2S_BCLK = 26, PIN_I2S_LRC = 25, PIN_I2S_DOUT = 27;           // MAX98357A и PCM5102A
const int PIN_AMP_SD = 4;                                                   // вывод SD усилителя: глушит вибродинамик
const int PIN_LED = 13;                                                     // WS2812: кольцо, затем 2 диода ручки
const int PIN_ENC_A = 16, PIN_ENC_B = 17, PIN_ENC_SW = 14;                  // энкодер EC11

// ---------------------------------------------------------------- настройки
const int   RING_N = 24;             // диодов в кольце
const int   KNOB_N = 2;              // диодов подсветки ручки (в цепочке после кольца)
const int   RING_DIR = 1;            // 1 или -1: поменяйте, если свет бежит против часовой стрелки
const float RING_RPS = 33.333f / 60; // обороты в секунду, как у винила
const uint8_t LED_BRIGHT = 60;       // общая яркость 0..255
const int   VOL_STEPS = 20;          // шагов громкости
const float VOL_MAX_GAIN = 1.0f;     // усиление на максимуме (ESP8266Audio: 0..4)
const uint32_t NFC_PERIOD_MS = 400;  // опрос метки
const int   NFC_MISSES = 2;          // столько пропусков подряд = пластинку сняли
const CRGB  COLOR_DEFAULT = CRGB(255, 150, 60);

// ---------------------------------------------------------------- состояние
enum State { IDLE, PLAYING, PAUSED, UNKNOWN_TAG };
State state = IDLE;

struct Tag { String uid, path; CRGB color; };
std::vector<Tag> tags;
std::vector<String> playlist;        // треки текущей пластинки
int trackIdx = 0;
String currentUid;                   // пластинка, чья музыка загружена
CRGB currentColor = COLOR_DEFAULT;
uint32_t resumePos = 0;              // позиция в файле на паузе
bool userPaused = false;             // пауза нажатием ручки (пластинка лежит)
bool auxMode = false;                // true: звук только в AUX, вибродинамик выключен
int volume = 10;

CRGB leds[RING_N + KNOB_N];
Preferences prefs;
Adafruit_PN532 nfc(PIN_NFC_IRQ, PIN_NFC_RST, &Wire);

AudioFileSourceSD *srcFile = nullptr;
AudioFileSourceID3 *srcId3 = nullptr;
AudioGeneratorMP3 *mp3 = nullptr;
AudioOutputI2S *out = nullptr;

// метка: пишет задача NFC (ядро 0), читает loop (ядро 1)
portMUX_TYPE nfcMux = portMUX_INITIALIZER_UNLOCKED;
volatile bool nfcPresent = false;
char nfcUid[24] = "";

// ---------------------------------------------------------------- NFC в отдельной задаче
void nfcTask(void *) {
  int misses = NFC_MISSES;
  for (;;) {
    uint8_t uid[8]; uint8_t len = 0;
    bool ok = nfc.readPassiveTargetID(PN532_MIFARE_ISO14443A, uid, &len, 120);
    if (ok && len > 0) {
      char buf[24] = "";
      for (int i = 0; i < len && i < 10; i++) sprintf(buf + 2 * i, "%02X", uid[i]);
      portENTER_CRITICAL(&nfcMux); strcpy(nfcUid, buf); nfcPresent = true; portEXIT_CRITICAL(&nfcMux);
      misses = 0;
    } else if (++misses >= NFC_MISSES) {
      portENTER_CRITICAL(&nfcMux); nfcPresent = false; portEXIT_CRITICAL(&nfcMux);
    }
    vTaskDelay(pdMS_TO_TICKS(NFC_PERIOD_MS));
  }
}

// ---------------------------------------------------------------- карта памяти
CRGB parseColor(const String &s) {
  if (s.length() != 6) return COLOR_DEFAULT;
  long v = strtol(s.c_str(), nullptr, 16);
  return CRGB((v >> 16) & 255, (v >> 8) & 255, v & 255);
}

void loadTags() {
  tags.clear();
  File f = SD.open("/tags.csv");
  if (!f) { Serial.println("нет /tags.csv"); return; }
  while (f.available()) {
    String line = f.readStringUntil('\n'); line.trim();
    if (line.isEmpty() || line[0] == '#') continue;
    int a = line.indexOf(';'); if (a < 0) continue;
    int b = line.indexOf(';', a + 1);
    Tag t;
    t.uid = line.substring(0, a); t.uid.trim(); t.uid.toUpperCase();
    t.path = (b < 0 ? line.substring(a + 1) : line.substring(a + 1, b)); t.path.trim();
    String c = (b < 0 ? String("") : line.substring(b + 1)); c.trim();
    t.color = parseColor(c);
    if (!t.path.startsWith("/")) t.path = "/" + t.path;
    tags.push_back(t);
  }
  f.close();
  Serial.printf("пластинок в таблице: %d\n", (int)tags.size());
}

bool isMp3(String name) { name.toLowerCase(); return name.endsWith(".mp3"); }

void buildPlaylist(const String &path) {
  playlist.clear();
  File f = SD.open(path);
  if (!f) return;
  if (!f.isDirectory()) { f.close(); if (isMp3(path)) playlist.push_back(path); return; }
  for (File e = f.openNextFile(); e; e = f.openNextFile()) {
    String n = e.name();
    if (!e.isDirectory() && isMp3(n)) {
      int slash = n.lastIndexOf('/');
      playlist.push_back(path + "/" + n.substring(slash + 1));
    }
    e.close();
  }
  f.close();
  std::sort(playlist.begin(), playlist.end(), [](const String &a, const String &b) { return a.compareTo(b) < 0; });
}

void logUnknown(const String &uid) {
  Serial.printf("незнакомая пластинка: %s\n", uid.c_str());
  File f = SD.open("/unknown.txt", FILE_APPEND);
  if (f) { f.println(uid); f.close(); }
}

// ---------------------------------------------------------------- звук
void applyVolume() {
  float g = VOL_MAX_GAIN * powf((float)volume / VOL_STEPS, 2.0f);   // квадратичная кривая: ровнее на слух
  if (out) out->SetGain(g);
}

void applyOutputMode() {
  if (auxMode) { pinMode(PIN_AMP_SD, OUTPUT); digitalWrite(PIN_AMP_SD, LOW); }   // усилитель выключен
  else pinMode(PIN_AMP_SD, INPUT);                                               // усилитель работает
}

void stopAudio() {
  if (mp3) { if (mp3->isRunning()) mp3->stop(); delete mp3; mp3 = nullptr; }
  if (srcId3) { delete srcId3; srcId3 = nullptr; }
  if (srcFile) { delete srcFile; srcFile = nullptr; }
}

bool startTrack(uint32_t pos = 0) {
  stopAudio();
  if (trackIdx < 0 || trackIdx >= (int)playlist.size()) return false;
  srcFile = new AudioFileSourceSD(playlist[trackIdx].c_str());
  if (!srcFile->isOpen()) { Serial.printf("не открыть %s\n", playlist[trackIdx].c_str()); stopAudio(); return false; }
  mp3 = new AudioGeneratorMP3();
  bool ok;
  if (pos > 0) { srcFile->seek(pos, SEEK_SET); ok = mp3->begin(srcFile, out); }
  else { srcId3 = new AudioFileSourceID3(srcFile); ok = mp3->begin(srcId3, out); }
  applyVolume();
  Serial.printf("играет %s (с %u)\n", playlist[trackIdx].c_str(), (unsigned)pos);
  return ok;
}

void pauseAudio() {
  resumePos = srcFile ? srcFile->getPos() : 0;
  stopAudio();
  state = PAUSED;
}

// ---------------------------------------------------------------- ручка
volatile int encDelta = 0;
void IRAM_ATTR encIsr() {
  static uint8_t prev = 0; static int8_t acc = 0;
  static const int8_t tbl[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
  uint8_t cur = (digitalRead(PIN_ENC_A) << 1) | digitalRead(PIN_ENC_B);
  acc += tbl[(prev << 2) | cur]; prev = cur;
  if (acc >= 4) { encDelta++; acc = 0; } else if (acc <= -4) { encDelta--; acc = 0; }
}

enum Press { NONE, CLICK, DOUBLE, LONG };
Press readButton() {
  static bool down = false, longSent = false; static uint32_t tDown = 0, tUp = 0; static int clicks = 0;
  bool now = digitalRead(PIN_ENC_SW) == LOW; uint32_t t = millis();
  if (now && !down) { down = true; tDown = t; longSent = false; }
  if (down && now && !longSent && t - tDown > 800) { longSent = true; clicks = 0; return LONG; }
  if (!now && down) { down = false; if (!longSent && t - tDown > 25) { clicks++; tUp = t; } }
  if (clicks > 0 && !down && t - tUp > 300) { Press p = clicks >= 2 ? DOUBLE : CLICK; clicks = 0; return p; }
  return NONE;
}

// ---------------------------------------------------------------- свет
float ringPos = 0, ringSpeed = 0;     // положение (в диодах) и скорость (0..1 от полной)
uint32_t flashUntil = 0;

void drawLeds() {
  static uint32_t last = 0; uint32_t t = millis();
  if (t - last < 20) return;
  float dt = (t - last) / 1000.0f; last = t;
  float target = (state == PLAYING) ? 1.0f : 0.0f;
  ringSpeed += (target - ringSpeed) * min(1.0f, dt * 1.6f);            // плавный разгон и остановка, как у винила
  ringPos = fmodf(ringPos + RING_DIR * ringSpeed * RING_RPS * RING_N * dt + RING_N, RING_N);

  if (t < flashUntil) {                                                // незнакомая пластинка: красные вспышки
    bool on = ((flashUntil - t) / 200) % 2 == 0;
    fill_solid(leds, RING_N, on ? CRGB(180, 0, 0) : CRGB::Black);
  } else if (ringSpeed > 0.02f) {                                      // «комета» с хвостом
    const float tail = RING_N * 0.45f;
    for (int i = 0; i < RING_N; i++) {
      float d = fmodf((ringPos - i) * RING_DIR + RING_N * 2, RING_N);  // расстояние позади головы
      float k = d < tail ? powf(1.0f - d / tail, 2.2f) : 0.0f;
      float head = max(0.0f, 1.0f - fabsf(d < RING_N / 2 ? d : d - RING_N));   // сглаживание головы между диодами
      k = max(k, head) * (0.25f + 0.75f * ringSpeed);
      leds[i] = CRGB(currentColor.r * k, currentColor.g * k, currentColor.b * k);
    }
  } else {                                                             // ожидание: медленное «дыхание»
    float b = 0.06f + 0.10f * (0.5f + 0.5f * sinf(t / 1400.0f));
    CRGB c = (state == PAUSED) ? currentColor : COLOR_DEFAULT;
    fill_solid(leds, RING_N, CRGB(c.r * b, c.g * b, c.b * b));
  }
  float kv = 0.15f + 0.85f * volume / VOL_STEPS;                       // ручка: ярче при большей громкости
  CRGB kc = auxMode ? CRGB(40, 90, 255) : CRGB(255, 170, 80);
  for (int i = 0; i < KNOB_N; i++) leds[RING_N + i] = CRGB(kc.r * kv, kc.g * kv, kc.b * kv);
  FastLED.show();
}

// ---------------------------------------------------------------- логика пластинки
void onDiscPlaced(const String &uid) {
  if (uid == currentUid && state == PAUSED && !playlist.empty()) {     // вернули ту же пластинку
    if (!userPaused && startTrack(resumePos)) state = PLAYING;
    return;
  }
  for (auto &t : tags) if (t.uid == uid) {
    buildPlaylist(t.path);
    if (playlist.empty()) { Serial.printf("нет музыки по пути %s\n", t.path.c_str()); break; }
    currentUid = uid; currentColor = t.color; trackIdx = 0; userPaused = false;
    state = startTrack() ? PLAYING : IDLE;
    return;
  }
  stopAudio(); currentUid = ""; playlist.clear();
  logUnknown(uid);
  state = UNKNOWN_TAG; flashUntil = millis() + 1200;
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_OFF); btStop();
  pinMode(PIN_ENC_A, INPUT_PULLUP); pinMode(PIN_ENC_B, INPUT_PULLUP); pinMode(PIN_ENC_SW, INPUT_PULLUP);
  attachInterrupt(PIN_ENC_A, encIsr, CHANGE); attachInterrupt(PIN_ENC_B, encIsr, CHANGE);

  FastLED.addLeds<WS2812B, PIN_LED, GRB>(leds, RING_N + KNOB_N);
  FastLED.setBrightness(LED_BRIGHT);
  FastLED.setMaxPowerInVoltsAndMilliamps(5, 350);
  FastLED.clear(true);

  prefs.begin("minivinyl", false);
  volume = constrain(prefs.getInt("vol", 10), 0, VOL_STEPS);
  auxMode = prefs.getBool("aux", false);
  applyOutputMode();

  if (!SD.begin(PIN_SD_CS)) Serial.println("карта памяти не найдена");
  else loadTags();

  out = new AudioOutputI2S();
  out->SetPinout(PIN_I2S_BCLK, PIN_I2S_LRC, PIN_I2S_DOUT);
  applyVolume();

  Wire.begin(PIN_SDA, PIN_SCL);
  nfc.begin();
  if (!nfc.getFirmwareVersion()) Serial.println("PN532 не отвечает");
  nfc.SAMConfig();
  xTaskCreatePinnedToCore(nfcTask, "nfc", 4096, nullptr, 1, nullptr, 0);
}

void loop() {
  // звук: кормим декодер как можно чаще
  if (state == PLAYING && mp3) {
    if (!mp3->loop()) {                                                // трек кончился
      if (++trackIdx < (int)playlist.size() && startTrack()) { /* следующий трек альбома */ }
      else { stopAudio(); trackIdx = 0; resumePos = 0; userPaused = true; state = PAUSED; }
    }
  }

  // пластинка
  static bool wasPresent = false; static String lastUid;
  portENTER_CRITICAL(&nfcMux); bool present = nfcPresent; String uid = nfcUid; portEXIT_CRITICAL(&nfcMux);
  if (present && (!wasPresent || uid != lastUid)) onDiscPlaced(uid);
  if (!present && wasPresent) {
    if (state == PLAYING) pauseAudio();
    else if (state == UNKNOWN_TAG) state = IDLE;
    userPaused = false;
  }
  wasPresent = present; lastUid = uid;

  // ручка: поворот — громкость
  noInterrupts(); int d = encDelta; encDelta = 0; interrupts();
  if (d) {
    volume = constrain(volume + d, 0, VOL_STEPS);
    applyVolume(); prefs.putInt("vol", volume);
  }
  // ручка: нажатие — пауза, двойное — следующий трек, долгое — вибродинамик / AUX
  switch (readButton()) {
    case CLICK:
      if (state == PLAYING) { pauseAudio(); userPaused = true; }
      else if (state == PAUSED && present && !playlist.empty()) { userPaused = false; if (startTrack(resumePos)) state = PLAYING; }
      break;
    case DOUBLE:
      if ((state == PLAYING || state == PAUSED) && !playlist.empty()) {
        trackIdx = (trackIdx + 1) % playlist.size(); userPaused = false;
        state = startTrack() ? PLAYING : IDLE;
      }
      break;
    case LONG:
      auxMode = !auxMode; applyOutputMode(); prefs.putBool("aux", auxMode);
      break;
    default: break;
  }

  drawLeds();
}
