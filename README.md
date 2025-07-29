# CyberSec: Топологический и Геометрический Анализ ECDSA
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/579734f1-91e4-4699-bf55-68322a173c36" />


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**CyberSec** - это продвинутая система безопасности, разработанная для анализа и защиты реализаций цифровой подписи ECDSA (Elliptic Curve Digital Signature Algorithm). Основываясь на наших глубоких исследованиях в области топологии и геометрии эллиптических кривых, система способна обнаруживать тонкие уязвимости, связанные с генерацией случайных чисел (nonce), которые могут быть незаметны для традиционных методов.

**Наша цель - защита данных, а не их взлом.**
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/ffc0c256-029b-41d9-abc5-4cd6d48ebdef" />


## Особенности

CyberSec реализует комплексный подход к анализу безопасности ECDSA, используя:

*   **Топологический анализ:** Анализ распределения параметров подписи (`u_r`, `u_z`) в параметризующем пространстве с использованием чисел Бетти, топологической энтропии и расстояния Вассерштейна для выявления аномалий и предсказуемости.
*   **Градиентный анализ:** Оценка чувствительности подписи к изменению параметров для обнаружения утечек информации.
*   **Сдвиговые инварианты:** Поиск корреляций между подписями, указывающих на недостаточную независимость генерации nonce.
*   **DFT-анализ:** Спектральный анализ для выявления периодичностей в данных подписей.
*   **Анализ зеркальных пар:** Обнаружение структурных уязвимостей, связанных с определенными парами подписей.
*   **Усиленная генерация подписей:** Реализация безопасной генерации nonce на основе хэш-цепочек (RFC 6979 и собственные методы).
*   **Поддержка GPU/CPU:** Использование `CuPy` для ускорения вычислений на GPU (опционально).
*   **Работа с файлами:** Загрузка и анализ подписей из различных форматов (JSON, CSV).
*   **Автоматическая генерация отчетов:** Создание подробных отчетов об уязвимостях и рекомендациях по усилению безопасности.

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/ваш_логин/CyberSec.git
    cd CyberSec
    ```
2.  **Создайте виртуальное окружение (рекомендуется):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # На Windows используйте `venv\Scripts\activate`
    ```
3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Предполагается, что файл `requirements.txt` создан и содержит: `ecdsa`, `numpy`, `scipy`)*
4.  **(Опционально) Установите CuPy для поддержки GPU:**
    Следуйте официальной инструкции для установки `cupy`: https://docs.cupy.dev/en/stable/install.html
    Например, для CUDA 11.x:
    ```bash
    pip install cupy-cuda11x
    ```

## Использование

CyberSec может использоваться как библиотека в ваших Python-скриптах или как самостоятельное приложение с интерфейсом командной строки (CLI).

### 1. Использование как библиотеки:
___
```python
from CyberSec import CyberSec, SecurityLevel
___

# Создание экземпляра системы
cybersec = CyberSec(security_level=SecurityLevel.HIGH, use_gpu=True) # Использовать GPU, если доступно

# Генерация ключей
private_key, public_key = cybersec.generate_key_pair()

# Создание подписи
message = b"Ваше сообщение для подписи"
signature = cybersec.sign(message)

# Проверка подписи
is_valid = cybersec.verify(message, signature)
print(f"Подпись валидна: {is_valid}")

# Генерация искусственных подписей для анализа (или загрузка реальных)
artificial_signatures = cybersec.generate_artificial_signatures(1000)
# cybersec.load_signatures_from_file("path/to/signatures.json")

# Проведение топологического и комплексного анализа
security_report = cybersec.check_security()

# Получение и вывод результатов
vuln_report = cybersec.get_vulnerability_report()
print("Безопасность:", "Да" if vuln_report['analysis']['secure'] else "Нет")
if not vuln_report['analysis']['secure']:
    print("Обнаруженные проблемы:")
    for issue in vuln_report['analysis']['issues']:
        print(f" - {issue}")
    print("Рекомендации:")
    for rec in vuln_report['analysis']['recommendations']:
        print(f" - {rec}")

# Генерация отчета в файл
cybersec.generate_vulnerability_report_file("vulnerability_report.json")
# или
cybersec.generate_vulnerability_report_file("vulnerability_report.txt")
```

### 2. Использование CLI:

```bash
# Запуск демонстрационного режима
python CyberSec.py --debug

# Запуск с повышенным уровнем безопасности и GPU
python CyberSec.py --level MAXIMUM --gpu --debug

# Выполнение задачи из файла (см. пример файла задачи ниже)
python CyberSec.py --task my_task.json
```

**Пример файла задачи (`my_task.json`):**

```json
{
    "task": "protect_system",
    "signatures_file": "path/to/your_signatures.json",
    "report_file": "analysis_report.txt"
}
```

### Аргументы командной строки:

*   `--task TASK_FILE`: Путь к файлу JSON с задачей для выполнения.
*   `--level {BASIC,MEDIUM,HIGH,MAXIMUM}`: Уровень безопасности (по умолчанию: HIGH).
*   `--mode {PROTECT,AUDIT,VERIFY}`: Режим анализа (по умолчанию: PROTECT).
*   `--gpu`: Включить использование GPU ускорения (если доступно CuPy).
*   `--debug`: Включить режим отладки для подробного вывода.

## Лицензия

Этот проект лицензирован по лицензии MIT - подробности см. в файле [LICENSE](LICENSE).

## Контакты

[Алексей] - [@miroaleksej](https://github.com/miroaleksej) 
miro-aleksej@yandex.ru

Проект разработан как часть исследований в области криптографической безопасности.

```
