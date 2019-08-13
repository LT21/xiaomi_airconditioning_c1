"""
Support for Xiaomi Air Conditioning C1
"""
import enum
import logging
import asyncio

from collections import defaultdict
from typing import Optional
import click
from miio.device import Device, DeviceException
from miio.click_common import command, format_output, EnumType

from functools import partial
from datetime import timedelta
import voluptuous as vol

from homeassistant.core import callback
from homeassistant.components.climate import (
    ClimateDevice, PLATFORM_SCHEMA, )
from homeassistant.components.climate.const import (
    ATTR_HVAC_MODE, DOMAIN, HVAC_MODES, HVAC_MODE_OFF, HVAC_MODE_HEAT,
    HVAC_MODE_COOL, HVAC_MODE_AUTO, HVAC_MODE_DRY, HVAC_MODE_FAN_ONLY,
    SUPPORT_SWING_MODE, SUPPORT_FAN_MODE, SUPPORT_TARGET_TEMPERATURE, )
from homeassistant.const import (
    ATTR_ENTITY_ID, ATTR_TEMPERATURE, ATTR_UNIT_OF_MEASUREMENT, CONF_NAME,
    CONF_HOST, CONF_TOKEN, CONF_TIMEOUT, TEMP_CELSIUS, )
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers.event import async_track_state_change
import homeassistant.helpers.config_validation as cv
from homeassistant.util.dt import utcnow

_LOGGER = logging.getLogger(__name__)

SUCCESS = ['ok']

MODEL_AIRCONDITION_MA2 = 'xiaomi.aircondition.ma2'

MODELS_SUPPORTED = [MODEL_AIRCONDITION_MA2]

DEFAULT_NAME = 'Xiaomi Air Conditioning C1'
DATA_KEY = 'climate.xiaomi_airconditioning_c1'

CONF_MIN_TEMP = 'min_temp'
CONF_MAX_TEMP = 'max_temp'

ATTR_SWING_MODE = 'swing_mode'
ATTR_FAN_MODE = 'fan_mode'
ATTR_WIND_LEVEL = 'wind_level'

SCAN_INTERVAL = timedelta(seconds=15)

SUPPORT_FLAGS = (SUPPORT_TARGET_TEMPERATURE |
                 SUPPORT_FAN_MODE |
                 SUPPORT_SWING_MODE)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Required(CONF_TOKEN): vol.All(cv.string, vol.Length(min=32, max=32)),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_MIN_TEMP, default=16): vol.Coerce(int),
    vol.Optional(CONF_MAX_TEMP, default=30): vol.Coerce(int),
})

# pylint: disable=unused-argument
@asyncio.coroutine
def async_setup_platform(hass, config, async_add_devices, discovery_info=None):
    """Set up the air condition companion from config."""
    if DATA_KEY not in hass.data:
        hass.data[DATA_KEY] = {}

    host = config.get(CONF_HOST)
    token = config.get(CONF_TOKEN)
    name = config.get(CONF_NAME)
    min_temp = config.get(CONF_MIN_TEMP)
    max_temp = config.get(CONF_MAX_TEMP)

    _LOGGER.info("Initializing with host %s (token %s...)", host, token[:5])

    try:
        device = AirCondition(host, token)
        device_info = device.info()
        model = device_info.model
        unique_id = "{}-{}".format(model, device_info.mac_address)
        _LOGGER.info("model[ %s ] firmware_ver[ %s ] hardware_ver[ %s ] detected",
                     model,
                     device_info.firmware_version,
                     device_info.hardware_version)
    except DeviceException as ex:
        _LOGGER.error("Device unavailable or token incorrect: %s", ex)
        raise PlatformNotReady

    air_condition_companion = XiaomiAirCondition(
        hass, name, device, unique_id, min_temp, max_temp)
    hass.data[DATA_KEY][host] = air_condition_companion
    async_add_devices([air_condition_companion], update_before_add=True)


class HVACMode(enum.Enum):
    Off = HVAC_MODE_OFF
    Cool = HVAC_MODE_COOL
    Dry = HVAC_MODE_DRY
    Fan_Only = HVAC_MODE_FAN_ONLY
    Heat = HVAC_MODE_HEAT


class XiaomiAirCondition(ClimateDevice):
    """Representation of a Xiaomi Air Condition Companion."""

    def __init__(self, hass, name, device, unique_id,
                 min_temp, max_temp):

        """Initialize the climate device."""
        self.hass = hass
        self._name = name
        self._device = device
        self._unique_id = unique_id

        self._available = False
        self._state = None
        self._state_attrs = {
            ATTR_TEMPERATURE: None,
            ATTR_SWING_MODE: None,
            ATTR_HVAC_MODE: None,
        }

        self._max_temp = max_temp
        self._min_temp = min_temp
        self._current_temperature = None
        self._swing_mode = None
        self._wind_level = None
        self._hvac_mode = None
        self._target_temperature = 26

    @asyncio.coroutine
    def _try_command(self, mask_error, func, *args, **kwargs):
        """Call a command handling error messages."""
        try:
            result = yield from self.hass.async_add_job(
                partial(func, *args, **kwargs))

            _LOGGER.debug("Response received: %s", result)

            return result == SUCCESS
        except DeviceException as exc:
            _LOGGER.error(mask_error, exc)
            self._available = False
            return False

    @asyncio.coroutine
    def async_turn_on(self, speed: str = None, **kwargs) -> None:
        """Turn the miio device on."""
        result = yield from self._try_command(
            "Turning the miio device on failed.", self._device.on)

        if result:
            self._state = True

    @asyncio.coroutine
    def async_turn_off(self, **kwargs) -> None:
        """Turn the miio device off."""
        result = yield from self._try_command(
            "Turning the miio device off failed.", self._device.off)

        if result:
            self._state = False

    @asyncio.coroutine
    def async_update(self):
        """Update the state of this climate device."""
        try:
            state = yield from self.hass.async_add_job(self._device.status)
            _LOGGER.debug("new state: %s", state)

            self._available = True
            self._last_on_operation = HVACMode[state.mode.name].value
            if state.power == 0:
                self._hvac_mode = HVAC_MODE_OFF
                self._state = False
            else:
                self._hvac_mode = self._last_on_operation
                self._state = True
            self._current_temperature = state.temperature
            self._fan_mode = FanSpeed(state.wind_level).name
            self._swing_mode = SwingMode(state.swing).name
            self._state_attrs.update({
                ATTR_TEMPERATURE: self._target_temperature,
                ATTR_SWING_MODE: state.swing,
                ATTR_FAN_MODE: state.wind_level,
                ATTR_HVAC_MODE: state.mode.name.lower() if self._state else "off"
            })


        except DeviceException as ex:
            self._available = False
            _LOGGER.error("Got exception while fetching the state: %s", ex)

    @property
    def supported_features(self):
        """Return the list of supported features."""
        return SUPPORT_FLAGS

    @property
    def min_temp(self):
        """Return the minimum temperature."""
        return self._min_temp

    @property
    def max_temp(self):
        """Return the maximum temperature."""
        return self._max_temp

    @property
    def should_poll(self):
        """Return the polling state."""
        return True

    @property
    def unique_id(self):
        """Return an unique ID."""
        return self._unique_id

    @property
    def name(self):
        """Return the name of the climate device."""
        return self._name

    @property
    def available(self):
        """Return true when state is known."""
        return self._available

    @property
    def temperature_unit(self):
        """Return the unit of measurement."""
        return TEMP_CELSIUS

    @property
    def current_temperature(self):
        """Return the current temperature."""
        return self._current_temperature

    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        return self._target_temperature

    @property
    def last_on_operation(self):
        """Return the last operation when the AC is on (ie heat, cool, fan only)"""
        return self._last_on_operation

    @property
    def hvac_mode(self):
        """Return new hvac mode ie. heat, cool, fan only."""
        return self._hvac_mode

    @property
    def hvac_modes(self):
        """Return the list of available hvac modes."""
        return [mode.value for mode in HVACMode]

    @property
    def swing_mode(self):
        """Return the current swing setting."""
        return self._swing_mode

    @property
    def swing_modes(self):
        """List of available swing modes."""
        return [mode.name for mode in SwingMode]

    @property
    def fan_mode(self):
        """Return fan mode."""
        return self._fan_mode

    @property
    def fan_modes(self):
        """Return the list of available fan modes."""
        return [speed.name for speed in FanSpeed]

    @asyncio.coroutine
    def async_set_temperature(self, **kwargs):
        """Set target temperature."""
        if self._hvac_mode == HVAC_MODE_OFF or self._hvac_mode == HVAC_MODE_FAN_ONLY:
            return;

        if kwargs.get(ATTR_TEMPERATURE) is not None:
            self._target_temperature = kwargs.get(ATTR_TEMPERATURE)
        if kwargs.get(ATTR_HVAC_MODE) is not None:
            self._hvac_mode = OperationMode(kwargs.get(ATTR_HVAC_MODE))

        yield from self._try_command(
            "Setting temperature of the miio device failed.",
            self._device.set_temperature, self._target_temperature)

    @asyncio.coroutine
    def async_set_swing_mode(self, swing_mode):
        """Set the swing mode."""
        if self.supported_features & SUPPORT_SWING_MODE == 0:
            return

        self._swing_mode = SwingMode[swing_mode.title()].name

        yield from  self._try_command(
            "Setting swing mode of the miio device failed.",
            self._device.set_swing, self._swing_mode == SwingMode.On.name)

    @asyncio.coroutine
    def async_set_fan_mode(self, fan_mode):
        """Set the fan mode."""
        if self.supported_features & SUPPORT_FAN_MODE == 0:
            return

        if self._hvac_mode == HVAC_MODE_DRY:
            return

        self._fan_mode = FanSpeed[fan_mode.title()].name
        fan_mode_value = FanSpeed[fan_mode.title()].value

        yield from self._try_command(
            "Setting fan mode of the miio device failed.",
            self._device.set_wind_level, fan_mode_value)

    @asyncio.coroutine
    def async_set_hvac_mode(self, hvac_mode):
        """Set new target hvac mode."""
        if hvac_mode == HVAC_MODE_OFF:
            result = yield from self._try_command(
                "Turning the miio device off failed.", self._device.off)
            if result:
                self._state = False
                self._hvac_mode = HVAC_MODE_OFF
        else:
            if self._hvac_mode == HVAC_MODE_OFF:
                result = yield from self._try_command(
                "Turning the miio device on failed.", self._device.on)
            self._hvac_mode = HVACMode(hvac_mode).value
            self._state = True
            yield from self._try_command(
                "Setting hvac mode of the miio device failed.",
                self._device.set_mode, OperationMode[self._hvac_mode.title()])


class AirConditionException(DeviceException):
    pass


class OperationMode(enum.Enum):
    Cool = 2
    Dry = 3
    Fan_Only = 4
    Heat = 5


class FanSpeed(enum.Enum):
    Auto = 0
    Level_1 = 1
    Level_2 = 2
    Level_3 = 3
    Level_4 = 4
    Level_5 = 5
    Level_6 = 6
    Level_7 = 7


class SwingMode(enum.Enum):
    On = 1
    Off = 0


class AirConditionStatus:
    """Container for status reports of the Xiaomi Air Condition."""

    def __init__(self, data):
        """
        Device model: zhimi.aircondition.ma1
        {'power': 1,
         'is_on': True,
         'mode': 2,
         'temperature': 26.5,
         'swing': True,
         'wind_level': 0,
         'dry': True,
         'energysave': True,
         'sleep': True,
         'light': True,
         'beep': True,
         'timer': '0,0,0,0'}
        """
        self.data = data

    @property
    def power(self) -> int:
        """Current power state."""
        return self.data['power']

    @property
    def is_on(self) -> bool:
        """True if the device is turned on."""
        return self.power == 1

    @property
    def mode(self) -> Optional[OperationMode]:
        """Current operation mode."""
        try:
            return OperationMode(self.data['mode'])
        except TypeError:
            return None

    @property
    def temperature(self) -> float:
        """Current temperature."""
        return self.data['temperature']

    @property
    def swing(self) -> int:
        """Vertical swing."""
        return self.data['swing']

    @property
    def wind_level(self) -> int:
        """Wind level."""
        return self.data['wind_level']

    @property
    def dry(self) -> bool:
        """Dry mode"""
        return self.data["dry"] == 1

    @property
    def energysave(self) -> bool:
        """Energysave mode"""
        return self.data['energysave'] == 1

    @property
    def sleep(self) -> bool:
        """Sleep mode"""
        return self.data['sleep'] == 1

    @property
    def light(self) -> bool:
        """Light"""
        return self.data['light'] == 1

    @property
    def beep(self) -> bool:
        """Beep"""
        return self.data['beep'] == 1

    @property
    def timer(self) -> str:
        """Timer"""
        return self.data['timer']

    def __repr__(self) -> str:
        s = "<AirConditionStatus " \
            "power=%s, " \
            "is_on=%s, " \
            "mode=%s, " \
            "temperature=%s, " \
            "swing=%s, " \
            "wind level=%s, " \
            "dry=%s, " \
            "energysave=%s, " \
            "sleep=%s, " \
            "light=%s, " \
            "beep=%s, " \
            "timer=%s>" % \
            (self.power,
             self.is_on,
             self.mode,
             self.temperature,
             self.swing,
             self.wind_level,
             self.dry,
             self.energysave,
             self.sleep,
             self.light,
             self.beep,
             self.timer)
        return s

    def __json__(self):
        return self.data


class AirCondition(Device):

    def __init__(self, ip: str = None, token: str = None, model: str = MODEL_AIRCONDITION_MA2,
                 start_id: int = 0, debug: int = 0, lazy_discover: bool = True) -> None:
        super().__init__(ip, token, start_id, debug, lazy_discover)

        if model in MODELS_SUPPORTED:
            self.model = model
        else:
            self.model = MODEL_AIRCONDITION_MA2
            _LOGGER.debug("Device model %s unsupported. Falling back to %s.", model, self.model)

    @command(
        default_output=format_output(
            "",
            "Power: {result.power}\n"
            "Mode: {result.mode}\n"
            "Temperature: {result.temperature} °C\n"
            "Wind Level: {result.wind_level}\n"
        )
    )
    def status(self) -> AirConditionStatus:
        """Retrieve properties."""

        properties = [
            'power',
            'mode',
            'temperature',
            'swing',
            'wind_level',
            'dry',
            'energysave',
            'sleep',
            'light',
            'beep',
            'timer',
        ]

        # Something weird. A single request is limited to 1 property.
        # Therefore the properties are divided into multiple requests
        _props = properties.copy()
        values = []
        while _props:
            values.extend(self.send("get_prop", _props[:1]))
            _props[:] = _props[1:]

        properties_count = len(properties)
        values_count = len(values)
        if properties_count != values_count:
            _LOGGER.debug(
                "Count (%s) of requested properties does not match the "
                "count (%s) of received values.",
                properties_count, values_count)

        return AirConditionStatus(
            defaultdict(lambda: None, zip(properties, values)))

    @command(
        default_output=format_output("Powering the air condition on"),
    )
    def on(self):
        """Turn the air condition on."""
        return self.send("set_power", [1])

    @command(
        default_output=format_output("Powering the air condition off"),
    )
    def off(self):
        """Turn the air condition off."""
        return self.send("set_power", [0])

    @command(
        click.argument("temperature", type=float),
        default_output=format_output(
            "Setting target temperature to {temperature} degrees")
    )
    def set_temperature(self, temperature: float):
        """Set target temperature."""
        return self.send("set_temp", [temperature])

    @command(
        click.argument("wind_level", type=int),
        default_output=format_output(
            "Setting wind level to {wind_level}")
    )
    def set_wind_level(self, wind_level: int):
        """Set wind level."""
        if wind_level < 0 or wind_level > 7:
            raise AirConditionException("Invalid wind level level: %s", wind_level)

        return self.send("set_wind_level", [wind_level])

    @command(
        click.argument("swing", type=bool),
        default_output=format_output(
            lambda swing: "Turning on swing mode"
            if swing else "Turning off swing mode"
        )
    )
    def set_swing(self, swing: bool):
        """Set swing on/off."""
        if swing:
            return self.send("set_swing", [1])
        else:
            return self.send("set_swing", [0])

    @command(
        click.argument("dry", type=bool),
        default_output=format_output(
            lambda dry: "Turning on dry mode"
            if dry else "Turning off dry mode"
        )
    )
    def set_dry(self, dry: bool):
        """Set dry on/off."""
        if dry:
            return self.send("set_dry", [1])
        else:
            return self.send("set_dry", [0])

    @command(
        click.argument("energysave", type=bool),
        default_output=format_output(
            lambda energysave: "Turning on energysave mode"
            if energysave else "Turning off energysave mode"
        )
    )
    def set_energysave(self, energysave: bool):
        """Set energysave on/off."""
        if energysave:
            return self.send("set_energysave", [1])
        else:
            return self.send("set_energysave", [0])

    @command(
        click.argument("sleep", type=bool),
        default_output=format_output(
            lambda sleep: "Turning on sleep mode"
            if sleep else "Turning off sleep mode"
        )
    )
    def set_sleep(self, sleep: bool):
        """Set sleep on/off."""
        if sleep:
            return self.send("set_sleep", [1])
        else:
            return self.send("set_sleep", [0])

    
    @command(
        click.argument("mode", type=EnumType(OperationMode, False)),
        default_output=format_output("Setting operation mode to '{mode.value}'")
    )
    def set_mode(self, mode: OperationMode):
        """Set operation mode."""
        return self.send("set_mode", [mode.value])