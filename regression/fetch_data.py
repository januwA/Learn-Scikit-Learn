import ccxt
import json
import os

# 配置
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 500
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'btc_1h.json')

def fetch_btc_data():
    # 初始化币安交易所对象
    # 如果你在国内运行需要代理，取消下面的注释并配置代理地址
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'proxies': {
            'http': 'http://127.0.0.1:19996',
            'https': 'http://127.0.0.1:19996',
        },
    })

    print(f"正在从 Binance 获取 {SYMBOL} {TIMEFRAME} 数据 (数量: {LIMIT})...")
    
    try:
        # fetch_ohlcv 返回格式: [timestamp, open, high, low, close, volume]
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
        
        # 转换为更易读的字典列表格式
        data = []
        for candle in ohlcv:
            data.append({
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            })

        # 保存为 JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print(f"成功保存 {len(data)} 条数据至: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        print("\n提示：如果在国内运行，请检查网络设置或在代码中配置代理。")

if __name__ == "__main__":
    fetch_btc_data()
