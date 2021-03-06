<h2>Задание</h2>
<p>Детектирование “живого” голоса (антиспуфинг)</p> 
<h3>Постановка задачи</h3>
<p>Виртуальные ассистенты, устройства Internet-of-Things все больше входят в жизнь современного человека. Они не только помогают автоматизировать поисковые запросы, распознают лица и речь, выполняют простейшие команды, но и учатся вести мониторинг состояния здоровья, детектировать различные ситуации, информировать о важных для Пользователя событиях.</p>
<p>Для того, чтобы виртуальные ассистенты реагировали только на голос человека, присутствующего перед устройством, и не принимали во внимание речь из телевизора, радио, а также синтезированную, воспроизводимую роботами и другую, звучащую из динамиков, необходимы детекторы “живого” голоса.</p>
<p>Данная задача посвящена детектированию наличия “живого” голоса (класс 1) и его отделению от синтетического/конвертированного/перезаписанного голоса (класс 2).</p>
<p>Предлагается разработать систему с использованием элементов машинного обучения, которая обучается на заданной обучающей базе аудиозаписей и должна быть протестирована на тестовой базе аудиозаписей.</p>
<h3>Исходные данные</h3>
<ol>
<li>Обучающая база данных:
  <ul>
  <li>Датасет из 50000 wav файлов</li>
  <li>Файлы имеют метки human (класс №1) и spoof (класс №2)</li>
  </ul></li>
<li>Тестовая база данных:
  <ul>
  <li>Датасет из 5000 wav файлов</li>
  <li>База не имеет меток правильных ответов </li>
  </ul></li>
</ol>
<h3>Комментарии к реализации:</h3>
<p>За основу взято решение из этой <a href="https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2279.pdf">публикации</a></p>
<p>Использована только одна ResNet18, без формирования attention mask и передачи во второй экземпляр ResNet18.</p>
<p>Звуковые файлы приводятся к длительности 6 секунд и преобразуются в gd-gram’мы – временные развёртки времени групповой задержки.</p>
<p>В результате каждому файлу соответствует изображение 512x256, что соответствует 8 кГц x 6 секунд.</p>
