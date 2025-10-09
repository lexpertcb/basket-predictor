#!/bin/bash
# 🔁 Redémarrage BasketAI Ultra+ (simple & fiable)

echo "🧠 Vérification des processus Streamlit..."
PID=$(pgrep -f "streamlit run")
if [ ! -z "$PID" ]; then
  echo "🟥 Arrêt de Streamlit (PID: $PID)..."
  kill -9 $PID
  sleep 2
else
  echo "ℹ️ Aucun process Streamlit actif."
fi

echo "🟦 Démarrage BasketAI Ultra+ sur :8501..."
nohup streamlit run /root/basketai/basket-predictor/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  > /root/basketai/basket-predictor/streamlit.log 2>&1 &

sleep 3
if pgrep -f "streamlit run" >/dev/null; then
  echo "✅ En ligne : http://134.199.227.6:8501"
else
  echo "❌ Échec démarrage → log : /root/basketai/basket-predictor/streamlit.log"
fi
