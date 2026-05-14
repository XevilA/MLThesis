"""
Solar PV Generation Model (Section 3.2)
Simulates distributed generation using pvlib with Rayong TMY data.
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

from config import SolarParams, LATITUDE, LONGITUDE, ALTITUDE, TIMEZONE, YEAR


def generate_synthetic_tmy(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic TMY-like irradiance and temperature data for Rayong.
    
    Calibrated against Solcast/PVGIS data for the Rayong-Chonburi corridor:
    - Annual GHI: ~1,700-1,850 kWh/m² (tropical, monsoon-affected)
    - Monsoon suppression June-October
    - Clear-sky peaks November-February
    """
    n = len(index)
    hours = index.hour
    day_of_year = index.dayofyear
    
    # Solar geometry for clear-sky envelope
    solar_noon_peak = 950  # W/m² peak clear-sky GHI at solar noon
    
    # Seasonal cloud cover factor (1.0 = clear, lower = cloudy)
    # Monsoon: Jun(6)–Oct(10) heavy cloud; Dry: Nov(11)–Feb(2) clear
    month = index.month
    seasonal_cloud = np.ones(n)
    seasonal_cloud[np.isin(month, [6, 7, 8, 9])] = 0.55    # Heavy monsoon
    seasonal_cloud[np.isin(month, [10])] = 0.65              # Late monsoon
    seasonal_cloud[np.isin(month, [5])] = 0.75               # Pre-monsoon
    seasonal_cloud[np.isin(month, [3, 4])] = 0.80            # Hot season (some haze)
    seasonal_cloud[np.isin(month, [11, 12, 1, 2])] = 0.90   # Dry season

    # Hour-of-day solar profile (simplified bell curve for tropical latitude)
    sunrise, sunset = 6, 18  # approximate for ~12.7°N
    hour_frac = hours.values.astype(float)
    solar_profile = np.zeros(n)
    daytime = (hour_frac >= sunrise) & (hour_frac <= sunset)
    solar_angle = np.pi * (hour_frac[daytime] - sunrise) / (sunset - sunrise)
    solar_profile[daytime] = np.sin(solar_angle)

    # Clear-sky GHI
    ghi_clearsky = solar_noon_peak * solar_profile

    # Apply seasonal cloud cover with daily stochastic variation
    n_days = int(np.ceil(n / 24))
    daily_cloud_noise = rng.beta(5, 2, size=n_days)  # Skewed toward clear
    daily_cloud = np.repeat(daily_cloud_noise, 24)[:n]
    
    # Combine seasonal and daily cloud effects
    cloud_factor = seasonal_cloud * (0.5 + 0.5 * daily_cloud)
    
    # Add hourly stochastic cloud transients
    hourly_noise = rng.normal(0, 0.08, size=n)
    cloud_factor = np.clip(cloud_factor + hourly_noise, 0.1, 1.0)
    
    ghi = ghi_clearsky * cloud_factor
    ghi = np.maximum(ghi, 0)

    # DNI and DHI estimation using Erbs decomposition (simplified)
    kt = np.zeros(n)  # clearness index
    kt[ghi_clearsky > 0] = ghi[ghi_clearsky > 0] / ghi_clearsky[ghi_clearsky > 0]
    kt = np.clip(kt, 0, 1)
    
    # Erbs diffuse fraction model
    kd = np.where(kt <= 0.22, 1.0 - 0.09 * kt,
         np.where(kt <= 0.80, 0.9511 - 0.1604*kt + 4.388*kt**2 
                  - 16.638*kt**3 + 12.336*kt**4, 0.165))
    
    dhi = ghi * kd
    dni = np.where(solar_profile > 0.05, (ghi - dhi) / np.maximum(solar_profile, 0.05), 0)
    dni = np.maximum(dni, 0)

    # Ambient temperature: tropical profile for Rayong
    # Annual mean ~28°C, diurnal range ~6-8°C, monsoon slightly cooler
    T_annual_mean = 28.0
    seasonal_T = np.zeros(n)
    seasonal_T[np.isin(month, [12, 1, 2])] = -1.5    # Cool season
    seasonal_T[np.isin(month, [3, 4, 5])] = 2.5       # Hot season
    seasonal_T[np.isin(month, [6, 7, 8, 9, 10])] = 0.5  # Monsoon (humid, moderate)
    seasonal_T[np.isin(month, [11])] = -0.5
    
    diurnal_T = -3.5 * np.cos(2 * np.pi * (hour_frac - 14) / 24)  # Peak at 14:00
    daily_T_noise = np.repeat(rng.normal(0, 1.2, size=n_days), 24)[:n]
    hourly_T_noise = rng.normal(0, 0.5, size=n)
    
    temp_air = T_annual_mean + seasonal_T + diurnal_T + daily_T_noise + hourly_T_noise

    # Wind speed (for cell temperature model)
    wind_speed = rng.weibull(2.0, size=n) * 2.5 + 0.5  # m/s, light tropical winds

    weather = pd.DataFrame({
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "temp_air": temp_air,
        "wind_speed": wind_speed,
    }, index=index)

    return weather


def simulate_solar_pv(
    index: pd.DatetimeIndex,
    params: SolarParams,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate hourly solar PV generation using pvlib.
    Returns DataFrame with E_DG (kWh) and weather data.
    """
    # Generate synthetic weather data
    weather = generate_synthetic_tmy(index, rng)

    # Define location and PV system
    location = Location(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        tz=TIMEZONE,
        altitude=ALTITUDE,
    )

    # Get solar position
    solar_pos = location.get_solarposition(index)

    # POA irradiance (plane of array)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=params.tilt,
        surface_azimuth=params.azimuth,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=weather["dni"],
        ghi=weather["ghi"],
        dhi=weather["dhi"],
        model="isotropic",
    )

    # Cell temperature (Sandia model, open rack)
    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    cell_temp = pvlib.temperature.sapm_cell(
        poa_global=poa["poa_global"],
        temp_air=weather["temp_air"],
        wind_speed=weather["wind_speed"],
        a=temp_params["a"],
        b=temp_params["b"],
        deltaT=temp_params["deltaT"],
    )

    # DC power using simplified single-diode at reference conditions
    # P_dc = P_stc * (G/G_ref) * [1 + gamma * (T_cell - T_ref)]
    G_ref = 1000  # W/m² STC
    T_ref = 25    # °C STC
    gamma_pmp = -0.004  # %/°C temperature coefficient (typical poly-Si)

    effective_irradiance = poa["poa_global"].fillna(0).values
    P_dc = params.capacity_kWp * (effective_irradiance / G_ref) * (
        1 + gamma_pmp * (cell_temp.fillna(25).values - T_ref)
    )
    P_dc = np.maximum(P_dc, 0)

    # Apply DC losses
    dc_loss = 1 - (params.soiling_loss + params.mismatch_loss + params.wiring_dc_loss)
    P_dc *= dc_loss

    # Apply inverter efficiency and AC losses
    ac_loss = 1 - (params.inverter_standby_loss + params.transformer_loss)
    P_ac = P_dc * params.inverter_efficiency * ac_loss

    # Apply annual degradation (linear across simulation period)
    hours_elapsed = np.arange(len(index))
    degradation_factor = 1 - params.annual_degradation * (hours_elapsed / 8760)
    P_ac *= degradation_factor

    # Final solar generation in kWh (hourly resolution → kWh = kW × 1h)
    E_DG = np.maximum(P_ac, 0)

    result = pd.DataFrame({
        "E_DG": E_DG,
        "GHI": weather["ghi"].values,
        "T_amb": weather["temp_air"].values,
    }, index=index)

    return result


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    index = pd.date_range(f"{2024}-01-01", periods=8760, freq="1h", tz="Asia/Bangkok")
    params = SolarParams()
    solar = simulate_solar_pv(index, params, rng)
    print(f"Annual solar yield: {solar['E_DG'].sum()/1000:.1f} MWh")
    print(f"Peak hour: {solar['E_DG'].max():.1f} kWh")
    print(f"Capacity factor: {solar['E_DG'].sum()/(params.capacity_kWp*8760)*100:.1f}%")
    print(f"Mean temp: {solar['T_amb'].mean():.1f}°C")
    print(f"Annual GHI: {solar['GHI'].sum()/1000:.0f} kWh/m²")
