# /root/basketai/basket-predictor/restart.sh
#!/bin/bash
set -e

APP_DIR="/root/basketai/basket-predictor"
LOG="$APP_DIR/streamlit.log"

echo "🧠 Vérification des processus Streamlit..."
PID=$(pgrep -f "streamlit run" || true)
if [ -n "$PID" ]; then
  echo "🟥 Stop Streamlit (PID: $PID)"
  kill -9 "$PID" || true
  sleep 1
fi

echo "🔐 Vérification clé API..."
STATUS=$(curl -s -H "x-apisports-key: ${APISPORTS_KEY:-bb3ba63a8bfab1020390fe28bd180522}" https://v1.basketball.api-sports.io/status | grep -o '"active":\(true\|false\)')
echo "Status API: $STATUS"

echo "🚀 Démarrage BasketAI Ultra..."
nohup streamlit run "$APP_DIR/app.py" --server.address 0.0.0.0 --server.port 8501 > "$LOG" 2>&1 &

sleep 2
if pgrep -f "streamlit run" >/dev/null; then
  echo "✅ En ligne : http://134.199.227.6:8501"
else
  echo "❌ Échec (voir $LOG)"
fi
