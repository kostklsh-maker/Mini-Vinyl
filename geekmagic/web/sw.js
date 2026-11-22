// Минимальный service worker: нужен, чтобы страницу можно было установить
// на телефон и она появилась в меню «Поделиться». Ничего не кэширует.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {});
