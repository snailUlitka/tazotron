# TODO Checklist: как улучшить модель без новых пациентов

- [ ] Перейти на pretrained backbone (ImageNet): `ResNet34/50` или `ConvNeXt-Tiny`.
- [ ] Держать размер модели умеренным, чтобы не переобучаться на текущем объеме (~944 XRays).
- [ ] Усилить аугментации для DRR: rotation, translation, scale, brightness/contrast, gamma, noise, blur, cutout.
- [ ] Перегенерировать синты из тех же CT с несколькими вариантами на кейс.
- [ ] Варировать camera pose при рендере.
- [ ] Варировать seed некроза.
- [ ] Варировать intensity/объем некроза.
- [ ] Делать split по `case_id` (group split), а не по отдельным `.pt`, чтобы убрать leakage.
- [ ] Обучать ансамбль по фолдам и усреднять предсказания на инференсе.
- [ ] Добавить TTA на инференсе и усреднение предсказаний.
- [ ] Сравнивать с baseline из `reports/after_full_dataset.json` и оставлять только изменения, которые улучшают test F1/accuracy.
