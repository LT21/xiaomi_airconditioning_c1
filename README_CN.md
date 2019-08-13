# Xiaomi Air Conditioning C1 Component
这是一个用于支持米家互联网空调C1(`xiaomi.aircondition.ma2`)接入HomeAssistant的自定义插件。

![](device.jpg)

## [README of English][readme]

## 配置设备
修改 `configuration.yaml`

```yaml
climate:
  - platform: xiaomi_airconditioning_c1
    name: Xiaomi Mijia Air Conditioning C1
    host: 192.168.10.10
    token: 8a57cbff6dd59e2effcb37e931bc68bd
    scan_interval: 60
```

## 支持特性
* 电源 (On, Off)
* 模式 (Cool, Dry, Fan Only, Heat)
* 风速 (Auto, Level 1, Level 2, Level 3, Level 4, Level 5, Level 6, Level 7)
* 扫风 (On, Off)
* 目标温度
* 当前温度

[readme]: https://github.com/LT21/xiaomi_airconditioning_c1
[readme_cn]: https://github.com/LT21/xiaomi_airconditioning_c1/blob/master/README_CN.md