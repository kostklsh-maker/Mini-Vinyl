#!/data/data/com.termux/files/usr/bin/bash
# Установка моста gmtv на Android-телефон (Termux) с автозапуском в фоне.
#
#   1. Поставьте из F-Droid: Termux и Termux:Boot (один раз откройте Termux:Boot).
#   2. В Termux:  pkg install git && git clone <ваш репозиторий> ~/Mini-Vinyl
#   3.            bash ~/Mini-Vinyl/geekmagic/termux/setup.sh
#
# После этого мост работает в фоне и поднимается сам после перезагрузки телефона.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$HOME/.gmtv.env"

echo "== пакеты"
pkg install -y python python-pillow >/dev/null

if [ ! -f "$ENV" ]; then
  echo "== ищу часы в сети Wi-Fi…"
  FOUND="$(python "$DIR/gmtv.py" find 2>/dev/null | head -1 | cut -f1 || true)"
  read -r -p "IP часов [${FOUND:-192.168.1.196}]: " IP
  IP="${IP:-${FOUND:-192.168.1.196}}"
  read -r -p "Показывать каждый новый скриншот? (y/N): " W
  WATCH=""
  if [ "$W" = "y" ] || [ "$W" = "Y" ]; then
    termux-setup-storage || true
    WATCH="$HOME/storage/dcim/Screenshots"
    [ -d "$WATCH" ] || WATCH="$HOME/storage/pictures/Screenshots"
  fi
  cat > "$ENV" <<EOF
export GMTV_DEVICE="$IP"
export GMTV_WATCH="$WATCH"
EOF
fi

echo "== автозапуск (Termux:Boot)"
mkdir -p "$HOME/.termux/boot"
cat > "$HOME/.termux/boot/gmtv" <<EOF
#!/data/data/com.termux/files/usr/bin/sh
termux-wake-lock
. "$ENV"
cd "$DIR"
pkill -f "gmtv.py" 2>/dev/null
nohup python gmtv.py serve --host 127.0.0.1 --port 8240 >> "$HOME/gmtv.log" 2>&1 &
if [ -n "\$GMTV_WATCH" ]; then
  nohup python gmtv.py watch "\$GMTV_WATCH" >> "$HOME/gmtv.log" 2>&1 &
fi
EOF
chmod +x "$HOME/.termux/boot/gmtv"
sh "$HOME/.termux/boot/gmtv"
sleep 1

. "$ENV"
echo
echo "Готово. Часы: $GMTV_DEVICE"
echo "1) Откройте в Chrome:  http://127.0.0.1:8240"
echo "2) Меню ⋮ → «Установить приложение» — появится «Поделиться → Экран GeekMagic»."
echo "3) Настройки Android → Приложения → Termux → Батарея → «Без ограничений»."
echo "Журнал: ~/gmtv.log   Остановить: pkill -f gmtv.py   Сменить IP: rm ~/.gmtv.env и снова setup.sh"
