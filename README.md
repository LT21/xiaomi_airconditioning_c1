# Xiaomi Air Conditioning C1 Component 
This is a custom component for home assistant to support the Xiaomi Mijia Air Conditioning C1 (`xiaomi.aircondition.ma2`).

![](device.jpg)

## [README of Chinese][readme_cn]

## Setup Device
Edit `configuration.yaml`

```yaml
climate:
  - platform: xiaomi_airconditioning_c1
    name: Xiaomi Mijia Air Conditioning C1
    host: 192.168.10.10
    token: 8a57cbff6dd59e2effcb37e931bc68bd
    scan_interval: 60
```

## Features
* Power (On, Off)
* Mode (Cool, Dry, Fan Only, Heat)
* Wind Level (Auto, Level 1, Level 2, Level 3, Level 4, Level 5, Level 6, Level 7)
* Swing Mode (On, Off)
* Target Temperature
* Current Temperature

[readme]: https://github.com/LT21/xiaomi_airconditioning_c1
[readme_cn]: https://github.com/LT21/xiaomi_airconditioning_c1/blob/master/README_CN.md