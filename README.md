# Специализации Яндекс Лицея
## «Machine Learning» весна 2024 - проектный этап

[ML intensive Yandex Academy spring 2024][intensive]

### Описание

> В этом соревновании нужно написать сверточную нейронную сеть для классификации изображений.
> Вам предоставлен датасет с изображениями рентгена грудной клетки, включая: изображения с COVID-19
> изображения с инфекциями, не связанными с COVID-19 (вирусная или бактериальная пневмония)
> изображения здоровой грудной клетки 
> Размер обучающей выборки (train) - 27 тысяч изображений, тестовой выборки (test) - 7 тысяч изображений.
> Для изображений из обучающей выборки также предоставлены точные маски сегментации легких.
> В данном домашнем задании нет никаких ограничений по архитектуре модели, но реализовывать в pytorch/tensorflow её надо самим и обучить с нуля только на данном датасете, без внешних данных.

Для проекта была выбрана модель SqueezeNeXt (~2M params) и феймворк PyTorch (& torchvision), также применялась аугментация данных. Подробнее можно увидеть в файле `project.ipynb` 

### Пояснения

- project.ipynb - Jupyter Notebook с кодом проекта и комментариями
- project.html - html версия для чтения проекта
- models/ - директория с обученными моделями

## Ссылки на ресурсы

Архитектура _SqueezeNeXt_:

- [medium] - статья на Medium
- [github] - пример реализации архитектуры на github
- [arxiv] - оригинальная статья на arxiv

## Исходники

Другие ссылки где посмотреть исходники

| Source | Link |
| ------ | ------ |
| YandexDisk | [https://disk.yandex.ru/d/PdQzdRuKim-lnQ][disk] |
| GitHub | [https://github.com/AndryMaster/ML_model_x-ray_images][git] |

## License
MIT


   [intensive]: <https://www.kaggle.com/competitions/ml-intensive-yandex-academy-spring-2024/overview>
   [disk]: <https://disk.yandex.ru/d/PdQzdRuKim-lnQ>
   [git]: <https://github.com/AndryMaster/ML_model_x-ray_images>
   
   [medium]: <https://sh-tsang.medium.com/reading-squeezenext-hardware-aware-neural-network-design-image-classification-3fc8d1d3f76>
   [github]: <https://github.com/osmr/imgclsmob/blob/c03fa67de3c9e454e9b6d35fe9cbb6b15c28fda7/pytorch/pytorchcv/models/squeezenext.py>
   [arxiv]: <https://arxiv.org/abs/1803.10615>
