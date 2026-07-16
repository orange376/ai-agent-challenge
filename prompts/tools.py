# tools.py
def get_weather(city):
    """模拟天气查询"""
    weather_db = {
        "北京": "晴天，35°C，湿度40%",
        "上海": "多云，32°C，湿度70%",
        "成都": "阴天，28°C，湿度60%"
    }
    return weather_db.get(city, "未知城市")

def get_attractions(city):
    """模拟景点查询"""
    attractions_db = {
        "北京": ["故宫", "长城", "颐和园"],
        "上海": ["外滩", "迪士尼", "东方明珠"],
        "成都": ["宽窄巷子", "大熊猫基地", "锦里"]
    }
    return attractions_db.get(city, [])