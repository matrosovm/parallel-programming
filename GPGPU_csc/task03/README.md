# Задание 3. Фрактал Мандельброта. Сумма чисел. Максимальный по сумме префикс.

[![Build Status](https://app.travis-ci.com/GPGPUCourse/GPGPUTasks2021.svg?branch=task03)](https://app.travis-ci.com/GPGPUCourse/GPGPUTasks2021)

0. Сделать fork проекта
1. Выполнить задания 3.0, 3.1, 3.2 и 3.3 ниже

Задание 3.0.
=========

Ознакомьтесь с реализацией A+B и какие используются вспомогательные библиотеки:

1. ```src/main_aplusb.cpp``` - основная часть:

 - ```gpu::chooseGPUDevice``` - если у вас одно OpenCL-устройство, то можете не обращать внимания, если же несколько - то посмотрите, в каком порядке они печатаются в консоль, и запускайте программу с единственным аргументом запуска - номером используемого устройства
 - ```#include "cl/aplusb_cl.h"``` - в прошлом задании исходники кернела подгружались в runtime из текстового файла с диска, это может быть неудобно, поэтому библиотека позволяет сконвертировать текстовый файл с исходником ```aplusb.cl``` в header ```aplusb_cl.h``` с массивом байт состоящим из байт исходного файла (см. подробнее в ```CMakeLists.txt:22```)

2. ```src/cl/aplusb.cl``` - исходник кернела, который на этапе компиляции сохраняется ввиде массива байт в ```src/cl/aplusb_cl.h```. Обратите внимание на дефайн ```WARP_SIZE``` - его выставляет библиотека в зависимости от того, что за устройство используется, см. подробнее ```libs/gpu/libgpu/opencl/engine.cpp:579``` и ```libs/gpu/libgpu/opencl/engine.cpp:198```. Это значение может быть удобно для уменьшения числа ```barrier(...)``` - выставлением лишь там, где нужна синхронизация между потоками из одной work group, но из разного warp/wavefront, т.к. потоки из одного warp/wavefront синхронизированы между собой на уровне железа (но, теоретически, это undefined behaviour)

3. ```CMakeLists.txt``` - OpenMP опционален и полезен для ускорения CPU-стороны, здесь указывается, какие файлы компилировать и с какими библиотеками их слинковывать. add_subdirectory(libs) указывает cmake на библиотеки

4. ```libs/clew``` - та же библиотека, что и в прошлых заданиях. Позволяет в runtime слинковаться с OpenCL драйвером

5. ```libs/gpu``` - обертка над основным функционалом OpenCL и CUDA, позволяющая прозрачно использовать оба API (прозрачно вплоть до вызова кернела)

6. ```libs/images``` - обертка над небольшой библиотекой [CImg](http://cimg.eu/), которая позволяет читать и сохранять картинки и создавать простые окна

7. ```libs/utils``` - вспомогательные вещи вроде быстрого генератора псевдослучайных чисел и секундомера

В этом задании нужно только ознакомиться со структурой.

Задание 3.1. Фрактал Мандельброта
=========

![Mandelbrot](/.figures/mandelbrot.png?raw=true)

Реализуйте расчет фрактала Мандельброта на OpenCL - см. ```src/main_mandelbrot.cpp``` и ```src/cl/mandelbrot.cl```.

Комментарии
-----------

Подробнее про то, что считается, можно прочитать на [wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set#Escape_time_algorithm).

Эта задача computation-bound, поэтому полученные гигафлопы должны быть довольно близки (например 50% - все еще довольно близко) к теоретическим гигафлопам вашего устройства.

Результат прогрузить в ```gpu_results```. Обратите внимание, что в ```images::Image``` данные из одной строки лежат в памяти подряд, а данные одного столбца - с шагом в размер каждой строки. См. использование ```gpu_results.ptr()```.

Если вы увидите это на Windows:

![CImg fail on Windows](/.figures/cimg_windows_fail.png)

То:
1. Скачайте файл ```GraphicsMagick-1.3.34-Q8-win64-dll.exe``` (отсюда - [ftp://ftp.graphicsmagick.org/pub/GraphicsMagick/windows/](ftp://ftp.graphicsmagick.org/pub/GraphicsMagick/windows/))
2. Установите и не забудьте перезапустить среду разработки (чтобы при выполнении программы ```PATH``` был обновленный после установки GraphicsMagick)

Задание 3.2. Суммирование чисел
==============

Реализуйте суммирование чисел на OpenCL - см. ```src/main_sum.cpp```.

Минимум нужно реализовать суммирование с использованием локальной памяти и суммированием по локальной памяти первым потоком из рабочей группы.

Максимум нужно реализовать суммирование бинарным деревом внутри рабочей группы и с минимальным числом барьеров за счет ```#if (WARP_SIZE > N)```. Еще потенциальные оптимизации - ручной unrolling циклов, обработка каждой рабочей группы больше чем по WorkGroupSize элементов и т.п..

Задание 3.3. Максимальный по сумме префикс
===============

Реализуйте поиск максимального по сумме префикса - см. ```src/main_max_prefix_sum```.

Максимальный по сумме префикс - такой префикс массива xs по j-ый элемент (исключительно), что сумма ```xs[0] + xs[1] + ... + xs[j - 1]``` максимальна.

Как и с суммированием, достаточно реализовать хоть что-то, ложащееся на модель массового параллелизма. Но здорово, если вы выставите минимальное число барьеров, сделаете unrolling и т.п..
