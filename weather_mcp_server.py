import os
import httpx

from mcp.server.fastmcp import FastMCP


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP("Weather MCP Server")


OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

BASE_URL = "https://api.openweathermap.org/data/2.5"


# ============================================================
# CURRENT WEATHER
# ============================================================

@mcp.tool()
async def get_current_weather(city: str) -> dict:
    """
    Get the current weather for a city.
    """

    if not OPENWEATHER_API_KEY:
        return {
            "error": "OPENWEATHER_API_KEY is not configured."
        }

    url = f"{BASE_URL}/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            url,
            params=params,
        )

    if response.status_code != 200:
        return {
            "error": f"OpenWeather API error: {response.status_code}",
            "details": response.text,
        }

    data = response.json()

    return {
        "city": data.get("name"),
        "country": data.get("sys", {}).get("country"),
        "temperature_c": data.get("main", {}).get("temp"),
        "feels_like_c": data.get("main", {}).get("feels_like"),
        "humidity_percent": data.get("main", {}).get("humidity"),
        "pressure_hpa": data.get("main", {}).get("pressure"),
        "weather": data.get("weather", [{}])[0].get("description"),
        "wind_speed_mps": data.get("wind", {}).get("speed"),
    }


# ============================================================
# WEATHER FORECAST
# ============================================================

@mcp.tool()
async def get_forecast(city: str) -> dict:
    """
    Get the 5-day weather forecast for a city.
    """

    if not OPENWEATHER_API_KEY:
        return {
            "error": "OPENWEATHER_API_KEY is not configured."
        }

    url = f"{BASE_URL}/forecast"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            url,
            params=params,
        )

    if response.status_code != 200:
        return {
            "error": f"OpenWeather API error: {response.status_code}",
            "details": response.text,
        }

    data = response.json()

    forecast_items = []

    for item in data.get("list", []):
        forecast_items.append(
            {
                "datetime": item.get("dt_txt"),
                "temperature_c": item.get("main", {}).get("temp"),
                "feels_like_c": item.get("main", {}).get("feels_like"),
                "humidity_percent": item.get("main", {}).get("humidity"),
                "weather": item.get("weather", [{}])[0].get(
                    "description"
                ),
                "wind_speed_mps": item.get("wind", {}).get("speed"),
            }
        )

    return {
        "city": data.get("city", {}).get("name"),
        "country": data.get("city", {}).get("country"),
        "forecast": forecast_items,
    }


# ============================================================
# START MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()