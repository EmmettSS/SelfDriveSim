# SelfDriveSim 🚗🤖

[![Tests](https://img.shields.io/badge/tests-144%20node%20%7C%2077%20pytest%20passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)]()
[![Django](https://img.shields.io/badge/django-channels%20%2B%20daphne-darkgreen.svg)]()
[![TensorFlow.js](https://img.shields.io/badge/tfjs-DAVE--2--GRU%20%2B%20COCO--SSD-orange.svg)]()
[![Three.js](https://img.shields.io/badge/three.js-3D%20Simulator-lightgrey.svg)]()

[English](#english) | [فارسی](#فارسی)

---

<a name="english"></a>
## English Documentation

### Overview & Architecture

**SelfDriveSim** is a closed-loop, hardware-in-the-loop (HIL) end-to-end autonomous driving simulator that runs entirely on everyday consumer hardware: a **PC**, a **mobile phone**, and a **local Wi-Fi network**.

* **On the PC**: A high-performance [Three.js](https://threejs.org/) 3D simulation renders a first-person three-lane highway with oncoming traffic, dynamic camera angles, collision physics, and telemetry.
* **On the Mobile Phone**: Mounted on a stand facing the PC monitor, the phone's browser captures the monitor through its rear camera.
* **In-Browser Deep Learning & Vision**: Without installing any mobile app or streaming raw video to external servers, an in-browser deep neural network ([TensorFlow.js](https://www.tensorflow.org/js)) processes the visual frames in real time and predicts 4 driving commands:
  - `steering` ($-1.0 \dots +1.0$)
  - `throttle` ($0.0 \dots 1.0$)
  - `brake` ($0.0 \dots 1.0$)
  - `turn_signal` ($0$: Off, $1$: Left, $2$: Right)
* **Real-Time Control Loop**: Control frames are transmitted over low-latency WebSockets via **Django Channels** and **Daphne ASGI** back to the PC simulator at 20 Hz, updating vehicle kinematics, hazard detection, and HUD telemetry.

```
 ┌───────────────────────── PC Monitor & Simulator ────────────────────────┐
 │                                                                         │
 │   Three.js 3D Highway Scene  <─── Control Update (20 Hz) ───────────┐   │
 │   • Kinematic ego vehicle physics                                   │   │
 │   • Computer vision obstacle detection & safety corridor evasion    │   │
 │   • Dynamic oncoming traffic & collision boundaries                 │   │
 │   • Expandable HUD, Autopilot Settings & Cinema View                │   │
 │        │ Telemetry & Hazard Feedback (4 Hz)                         │   │
 └────────┼────────────────────────────────────────────────────────────┼───┘
          │                                                            │
          │                   Local Wi-Fi Network                      │
          │               (Django Channels + Daphne)                   │
          ▼                                                            │
 ┌──────────────────────── Mobile Phone (Browser) ─────────────────────┼───┐
 │                                                                     │   │
 │   Rear Camera Video Stream (160x120 @ 20 FPS)                       │   │
 │        │                                                            │   │
 │        ├──> COCO-SSD Local Monitor Detection & 6-Point Calibration  │   │
 │        │    (Auto-detects & crops monitor frame / manual handle drag│   │
 │        │                                                            │   │
 │        ├──> Lens Occlusion & Finger Blocking Safety Protection      │   │
 │        │                                                            │   │
 │        └──> DAVE-2-GRU Convolutional-Recurrent Neural Network       │   │
 │             │ (Predicts steering, throttle, brake, turn signal)     │   │
 │             └───────────────────────────────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

### Key Features

1. **In-Browser DAVE-2-GRU Neural Autopilot**:
   - 3-frame temporal convolutional-recurrent network operating entirely client-side via TensorFlow.js.
   - Converted model shards loaded smoothly via `.dat` static files to bypass restrictive MIME web-server filters.
2. **COCO-SSD Local Monitor Auto-Lock & 6-Point Calibration**:
   - Zero-cloud, local quantized COCO-SSD object detection running in the mobile browser to automatically locate and lock onto TV/laptop monitor borders.
   - **6-Point Interactive Touch & Mouse Calibration**: Interactive drag handles allowing users to freely adjust monitor crop boundaries with clamping and reset capabilities.
3. **Lens Occlusion & Finger Block Safety Guard**:
   - Built-in visual variance and luminance monitor that detects when the phone lens is blocked or obscured, immediately disengaging autopilot to prevent erratic steering or runaway sway.
4. **Pure Computer Vision Obstacle Detection & Evasion**:
   - Lightweight, real-time computer vision engine directly in the simulator processing incoming video frames.
   - Geometric and chromatic filters rejecting false positives from asphalt shadows, lane divider stripes, and LCD subpixel chromatic aberrations.
   - **Continuous Throttle Lane Evasion**: When avoiding obstacles or changing lanes, ego vehicle maintains cruising throttle ($\ge 0.45$) to avoid stalling in the middle of highway traffic.
5. **Initial 10 km/h Smooth Start & 10s Spawn Protection**:
   - Ego vehicle launches smoothly at an initial cruising speed of 10 km/h.
   - Traffic generation algorithm keeps the player's initial lane clear for 10 seconds to allow seamless autopilot engagement.
6. **Expandable HUD & Autopilot Settings View**:
   - Redesigned visual test panel with dynamic aspect ratio and sizing modes: **1x** (Standard), **1.5x** (Large), and **Cinema ⤢** (Expanded 640px widescreen).
   - Side panel is freely resizable horizontally with live visual overlays showing detected obstacles and bounding boxes.
7. **One-Click Launch Scripts & Zero-Config HTTPS**:
   - `start.bat` for automatic virtual environment detection, IP resolution, and Daphne server startup.
   - `python manage.py runsecure` command providing automatic self-signed TLS certificates for local HTTPS, allowing mobile camera access on Chrome and Safari without manual flag configuration.

---

### Quick Start Guide

#### Prerequisites
- **Python 3.11** or **Python 3.12**
- **Modern Web Browser**: Google Chrome or Chromium-based browser on PC; Chrome or Safari on mobile.
- **Node.js 20+** *(optional, for running JS unit tests)*

#### 1. Setup & Installation
```bash
git clone https://github.com/EmmettSS/SelfDriveSim.git
cd SelfDriveSim/selfdriving_sim

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python manage.py migrate
```

#### 2. Running the Server

**Option A (One-Click Windows Script)**:
Double-click `start.bat` in the root folder or execute:
```cmd
start.bat
```

**Option B (Standard Secure ASGI Server)**:
```bash
python manage.py runsecure 0.0.0.0:8000
```

#### 3. Connecting Devices

1. **PC (Simulator)**:
   - Open `https://localhost:8000/` in Chrome and press `F11` for full screen.
2. **Mobile Phone (Autopilot Controller)**:
   - Connect your phone to the same Wi-Fi network.
   - Scan the QR code displayed in the PC terminal or visit `https://<PC-LAN-IP>:8000/dashboard/`.
   - Accept camera permissions. Use **Auto-Lock** or tap **تنظیم دستی کادر** (Manual Crop) to align the 6 handles with your monitor edges.
   - Tap **شروع خودران** (Start Autopilot).

---

### Testing & Verification

- **Node.js Frontend & Logic Tests**:
  ```bash
  cd selfdriving_sim
  npm test
  # 144 / 144 unit tests passing
  ```
- **Python Backend & Django Channels Tests**:
  ```bash
  cd selfdriving_sim
  .venv\Scripts\pytest client/tests simulation/tests
  # 77 passed, 2 skipped (Redis)
  ```

---

<a name="فارسی"></a>
## راهنمای فارسی (Farsi Documentation)

### مرور کلی و معماری سیستم

پروژه **SelfDriveSim** یک شبیه‌ساز مداربسته و سرتاسری (End-to-End) رانندگی خودران بر مبنای سخت‌افزار در حلقه (Hardware-in-the-Loop) است که به طور کامل روی سخت‌افزارهای استاندارد کاربری اجرا می‌شود: یک **رایانه شخصی (PC)**، یک **گوشی هوشمند** و یک **شبکه Wi-Fi محلی**.

* **روی رایانه (شبیه‌ساز)**: یک صحنه سه‌بعدی اول‌شخص با کتابخانه [Three.js](https://threejs.org/) اتوبانی سه‌بانده با ترافیک پویا، فیزیک سینماتیک حرکت خودرو، مرزبندی برخورد و نمایشگر اطلاعات پروازی (HUD) را رندر می‌کند.
* **روی گوشی موبایل**: گوشی روی پایه‌ای روبروی نمایشگر قرار گرفته و از طریق دوربین پشت، صفحه مانیتور را پایش می‌کند.
* **یادگیری عمیق در مرورگر**: بدون نیاز به نصب هرگونه اپلیکیشن بومی و بدون ارسال فریم‌های خام به سرور، یک شبکه عصبی پیچشی-تکرارشونده در بستر [TensorFlow.js](https://www.tensorflow.org/js) فریم‌های دوربین را بی‌درنگ پردازش کرده و ۴ فرمان رانندگی را پیش‌بینی می‌کند:
  - زاویه فرمان `steering` (از $-1.0$ تا $+1.0$)
  - گاز `throttle` (از $0.0$ تا $1.0$)
  - ترمز `brake` (از $0.0$ تا $1.0$)
  - راهنما `turn_signal` ($0$ خاموش، $1$ چپ، $2$ راست)
* **حلقه کنترل بلادرنگ**: داده‌های کنترلی با نرخ ۲۰ هرتز از طریق سوکت‌های وب **Django Channels** و وب‌سرور **Daphne ASGI** به شبیه‌ساز رایانه ارسال شده و وضعیت خودرو را به‌روز می‌کنند.

---

### قابلیت‌ها و ویژگی‌های کلیدی

1. **رانندگی خودکار با شبکه عصبی DAVE-2-GRU در مرورگر**:
   - پردازش توالی ۳ فریم اخیر به کمک شبکه کانولوشنی-تکرارشونده DAVE-2-GRU به صورت کاملاً کلاینت-ساید با TensorFlow.js.
   - تبدیل و بازخوانی ایمن وزن‌های مدل با پسوند `.dat` برای جلوگیری از خطاهای اعتبارسنجی MIME Type در سرورهای وب.
2. **تشخیص و قفل خودکار مانیتور با COCO-SSD و کالیبراسیون دستی ۶ نقطه‌ای**:
   - پیاده‌سازی مدل سبک و آفلاین COCO-SSD جهت شناسایی کادر مانیتور در فریم دوربین گوشی.
   - **کالیبراسیون دستی ۶ نقطه‌ای**: امکان تنظیم دقیق محدوده کراپ مانیتور با درگ ۶ دستگیره لمسی (۴ گوشه + ۲ نقطه میانی) روی موبایل و ماوس رایانه با امکان ریست به حالت خودکار.
3. **سیستم ایمنی تشخیص انسداد لنز (Lens Occlusion Guard)**:
   - پایش هوشمند تاری و پوشش لنز (نظیر قرار گرفتن انگشت جلوی لنز یا تاریکی شدید) و متوقف‌سازی ایمن فرمان‌ها جهت جلوگیری از انحراف و چرخش ناگهانی خودرو.
4. **سیستم بینایی ماشین خالص (Pure CV) برای شناسایی موانع و سبقت هوشمند**:
   - الگوریتم سبک پردازش تصویر در شبیه‌ساز جهت استخراج خودروها و موانع از فریم‌های ورودی دوربین.
   - فیلترهای هندسی و ابعادی برای حذف آلارم‌های کاذب ناشی از خطوط بریده جاده، سایه پل‌ها و انکسار رنگی ساب‌پیکسل‌های LCD.
   - **حفظ گاز در تغییر خط و مانور فرار**: حفظ فشار گاز ($\ge 0.45$) در طول تغییر خط برای جلوگیری از توقف کامل خودرو در وسط اتوبان.
5. **استارت نرم ۱۰ کیلومتر بر ساعت و منطقه امن ۱۰ ثانیه‌ای**:
   - شروع رانندگی با سرعت پایدار اولیه ۱۰ کیلومتر بر ساعت.
   - محافظت از لاین شروع به مدت ۱۰ ثانیه به گونه‌ای که هیچ خودرویی مستقیماً مقابل خودروی کاربر اسپاون نشود.
6. **پنل تنظیمات و دیدگاه دوربین گسترش‌پذیر (Expandable HUD)**:
   - بازطراحی بخش نمایش زنده دوربین در تنظیمات خلبان خودکار با حالت‌های اندازه: **1x** (استاندارد)، **1.5x** (بزرگ) و **سینمایی ⤢** (نمای گسترده ۶۴۰ پیکسلی).
   - قابلیت تغییر عرض پنل کناری به صورت درگ افقی و نمایش کادرهای تشخیصی موانع.
7. **اسکریپت اجرای سریع و گواهی امن HTTPS خودکار**:
   - اسکریپت `start.bat` برای راه‌اندازی با یک کلیک، تشخیص خودکار IP محلی و اجرای محیط مجازی.
   - دستور مدیریتی `python manage.py runsecure` با تولید خودکار گواهینامه‌های امنیتی Self-Signed جهت دسترسی به دوربین بدون نیاز به تنظیم پرچم‌های پیچیده کروم.

---

### راهنمای نصب و اجرای سریع

#### پیش‌نیازها
- **پایتون ۳.۱۱ یا ۳.۱۲**
- **مرورگر مدرن** (Chrome یا مرورگرهای مبتنی بر Chromium روی رایانه؛ Chrome یا Safari روی موبایل)
- **Node.js 20+** *(اختیاری، صرفاً جهت اجرای تست‌های کلاینت)*

#### ۱. نصب و راه‌اندازی محیط
```bash
git clone https://github.com/EmmettSS/SelfDriveSim.git
cd SelfDriveSim/selfdriving_sim

# ایجاد و فعال‌سازی محیط مجازی
python -m venv .venv
# در ویندوز:
.venv\Scripts\activate
# در لینوکس یا مک:
source .venv/bin/activate

# نصب وابستگی‌ها
pip install -r requirements.txt
python manage.py migrate
```

#### ۲. اجرای سرور

**روش اول (اجرای سریع در ویندوز با یک کلیک)**:
فایل `start.bat` در ریشه پروژه را باز کنید یا دستور زیر را وارد نمایید:
```cmd
start.bat
```

**روش دوم (اجرای مستقیم سرور امن ASGI)**:
```bash
python manage.py runsecure 0.0.0.0:8000
```

#### ۳. نحوه اتصال گوشی موبایل

1. **رایانه (شبیه‌ساز)**: آدرس `https://localhost:8000/` را در مرورگر باز کرده و کلید `F11` را برای تمام‌صفحه بزنید.
2. **گوشی هوشمند**: به همان شبکه وای‌فای مشترک متصل شوید، کد QR ظاهر شده در ترمینال را اسکن کرده یا آدرس `https://<PC-LAN-IP>:8000/dashboard/` را باز کنید.
3. دسترسی دوربین را تأیید کنید. در صورت نیاز با زدن دکمه **تنظیم دستی کادر** ۶ نقطه کادر مانیتور را مرتب کنید و در نهایت دکمه **شروع خودران** را لمس نمایید.

---

### تست‌ها و تضمین کیفیت نرم‌افزار

- **تست‌های واحد جاوااسکریپت (Node.js)**:
  ```bash
  cd selfdriving_sim
  npm test
  # ۱۴۴ تست با موفقیت پاس شدند (144 passed)
  ```
- **تست‌های پایتون و بک‌اند (Pytest)**:
  ```bash
  cd selfdriving_sim
  .venv\Scripts\pytest client/tests simulation/tests
  # ۷۷ تست با موفقیت پاس شدند (77 passed, 2 skipped)
  ```

---

## License

Copyright © 2026 EmmettSS. All rights reserved.

Designed & developed by **Emmett Software Development Group**.
