import psutil
import platform
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import openai
import numpy as np
from sklearn.ensemble import IsolationForest
import sqlite3

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage: Dict[str, float]
    network_stats: Dict[str, int]
    running_processes: int
    system_uptime: str
    temperature: Optional[float] = None

@dataclass
class AIDecision:
    action: str
    reasoning: str
    confidence: float
    suggested_commands: List[str]
    priority: str

class IntelligentSystemAgent:
    """Agente de IA inteligente para monitoreo y gestión de sistemas"""
    
    def __init__(self, config_file: str = 'ai_config.json'):
        self.config = self._load_config(config_file)
        self.db_path = 'system_metrics.db'
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        self.metrics_history = []
        
        # Configurar OpenAI (opcional)
        if self.config.get('openai_api_key'):
            openai.api_key = self.config['openai_api_key']
        
        self._init_database()
        self._load_historical_data()
        
        logger.info("🤖 Agente de IA inicializado")
    
    def _load_config(self, config_file: str) -> Dict:
        """Cargar configuración"""
        default_config = {
            "openai_api_key": "",  # Tu API key de OpenAI
            "slack_webhook": "",
            "learning_period_days": 7,
            "anomaly_threshold": 0.8,
            "auto_actions_enabled": False,
            "max_auto_actions_per_hour": 3
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"Config file not found. Creating default: {config_file}")
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _init_database(self):
        """Inicializar base de datos SQLite para almacenar métricas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage TEXT,
                anomaly_score REAL,
                actions_taken TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                situation TEXT,
                decision TEXT,
                reasoning TEXT,
                confidence REAL,
                outcome TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_historical_data(self):
        """Cargar datos históricos para entrenar el modelo de anomalías"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener últimos 7 días de datos
        week_ago = (datetime.now() - timedelta(days=self.config['learning_period_days'])).isoformat()
        cursor.execute(
            'SELECT cpu_percent, memory_percent FROM metrics WHERE timestamp > ? ORDER BY timestamp',
            (week_ago,)
        )
        
        data = cursor.fetchall()
        conn.close()
        
        if len(data) > 50:  # Necesitamos suficientes datos para entrenar
            self.metrics_history = np.array(data)
            self.anomaly_detector.fit(self.metrics_history)
            self.is_trained = True
            logger.info(f"🧠 Modelo entrenado con {len(data)} puntos de datos históricos")
        else:
            logger.info("📚 Recopilando datos para entrenar el modelo de anomalías...")
    
    def collect_metrics(self) -> SystemMetrics:
        """Recopilar métricas del sistema (igual que antes)"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = round((usage.used / usage.total) * 100, 2)
            except PermissionError:
                continue
        
        net_stats = psutil.net_io_counters()
        network_stats = {
            'bytes_sent': net_stats.bytes_sent,
            'bytes_recv': net_stats.bytes_recv
        }
        
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time).split('.')[0]
        
        temperature = None
        try:
            sensors = psutil.sensors_temperatures()
            if sensors:
                for name, entries in sensors.items():
                    if entries:
                        temperature = entries[0].current
                        break
        except:
            pass
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=round(memory.used / (1024**3), 2),
            memory_total_gb=round(memory.total / (1024**3), 2),
            disk_usage=disk_usage,
            network_stats=network_stats,
            running_processes=len(psutil.pids()),
            system_uptime=uptime,
            temperature=temperature
        )
    
    def detect_anomalies(self, metrics: SystemMetrics) -> float:
        """Detectar anomalías usando Machine Learning"""
        if not self.is_trained:
            return 0.0
        
        # Preparar datos para el modelo
        current_data = np.array([[metrics.cpu_percent, metrics.memory_percent]])
        
        # Calcular score de anomalía (-1 = anomalía, 1 = normal)
        anomaly_score = self.anomaly_detector.decision_function(current_data)[0]
        
        # Convertir a probabilidad (0-1, donde 1 = muy anómalo)
        anomaly_probability = max(0, (1 - anomaly_score) / 2)
        
        return anomaly_probability
    
    def analyze_situation_with_ai(self, metrics: SystemMetrics, anomaly_score: float) -> AIDecision:
        """Usar IA para analizar la situación y decidir acciones"""
        
        # Crear contexto detallado
        situation_context = f"""
        Sistema: {platform.node()}
        Timestamp: {metrics.timestamp}
        
        Métricas actuales:
        - CPU: {metrics.cpu_percent}%
        - Memoria: {metrics.memory_percent}% ({metrics.memory_used_gb}GB/{metrics.memory_total_gb}GB)
        - Procesos activos: {metrics.running_processes}
        - Uptime: {metrics.system_uptime}
        - Temperatura: {metrics.temperature}°C
        - Score de anomalía: {anomaly_score:.3f}
        
        Uso de disco:
        {json.dumps(metrics.disk_usage, indent=2)}
        """
        
        # Análisis con reglas inteligentes (sin OpenAI por simplicidad)
        return self._analyze_with_rules(metrics, anomaly_score, situation_context)
    
    def _analyze_with_rules(self, metrics: SystemMetrics, anomaly_score: float, context: str) -> AIDecision:
        """Análisis inteligente basado en reglas (alternativa a OpenAI)"""
        
        priority = "LOW"
        actions = []
        reasoning_parts = []
        
        # Análisis de CPU
        if metrics.cpu_percent > 90:
            priority = "CRITICAL"
            actions.extend([
                "ps aux --sort=-%cpu | head -10",
                "top -b -n1 | head -20",
                "systemctl status"
            ])
            reasoning_parts.append(f"CPU crítico al {metrics.cpu_percent}%")
        elif metrics.cpu_percent > 75:
            priority = "HIGH" if priority == "LOW" else priority
            actions.append("ps aux --sort=-%cpu | head -5")
            reasoning_parts.append(f"CPU elevado al {metrics.cpu_percent}%")
        
        # Análisis de memoria
        if metrics.memory_percent > 90:
            priority = "CRITICAL"
            actions.extend([
                "ps aux --sort=-%mem | head -10",
                "free -h",
                "sync && echo 1 > /proc/sys/vm/drop_caches"  # Liberar cache
            ])
            reasoning_parts.append(f"Memoria crítica al {metrics.memory_percent}%")
        elif metrics.memory_percent > 80:
            priority = "HIGH" if priority == "LOW" else priority
            actions.append("ps aux --sort=-%mem | head -5")
            reasoning_parts.append(f"Memoria elevada al {metrics.memory_percent}%")
        
        # Análisis de disco
        for mount, usage in metrics.disk_usage.items():
            if usage > 95:
                priority = "CRITICAL"
                actions.extend([
                    f"du -sh {mount}/* | sort -hr | head -10",
                    f"find {mount} -type f -size +100M",
                    "journalctl --disk-usage"
                ])
                reasoning_parts.append(f"Disco {mount} crítico al {usage}%")
        
        # Análisis de anomalías
        if anomaly_score > self.config['anomaly_threshold']:
            priority = "HIGH" if priority == "LOW" else priority
            actions.append("dmesg | tail -20")
            reasoning_parts.append(f"Comportamiento anómalo detectado (score: {anomaly_score:.3f})")
        
        # Determinar acción principal
        if priority == "CRITICAL":
            action = "IMMEDIATE_INTERVENTION"
        elif priority == "HIGH":
            action = "MONITOR_CLOSELY"
        else:
            action = "ROUTINE_CHECK"
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Sistema funcionando normalmente"
        confidence = min(0.9, 0.3 + (len(reasoning_parts) * 0.2))
        
        return AIDecision(
            action=action,
            reasoning=reasoning,
            confidence=confidence,
            suggested_commands=actions,
            priority=priority
        )
    
    def execute_ai_decision(self, decision: AIDecision, metrics: SystemMetrics):
        """Ejecutar decisiones del agente de IA"""
        logger.info(f"🤖 Decisión IA: {decision.action} (Confianza: {decision.confidence:.2f})")
        logger.info(f"💭 Razonamiento: {decision.reasoning}")
        
        # Guardar decisión en base de datos
        self._save_decision(decision, metrics)
        
        # Enviar notificación inteligente
        self._send_intelligent_notification(decision, metrics)
        
        # Ejecutar acciones automáticas si está habilitado
        if self.config['auto_actions_enabled'] and decision.priority in ['HIGH', 'CRITICAL']:
            self._execute_safe_commands(decision.suggested_commands[:2])  # Solo primeros 2 comandos
    
    def _save_decision(self, decision: AIDecision, metrics: SystemMetrics):
        """Guardar decisión en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO decisions (timestamp, situation, decision, reasoning, confidence, outcome)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp,
            f"CPU:{metrics.cpu_percent}% RAM:{metrics.memory_percent}%",
            decision.action,
            decision.reasoning,
            decision.confidence,
            "PENDING"
        ))
        
        # También guardar métricas
        cursor.execute('''
            INSERT INTO metrics (timestamp, cpu_percent, memory_percent, disk_usage, anomaly_score, actions_taken)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp,
            metrics.cpu_percent,
            metrics.memory_percent,
            json.dumps(metrics.disk_usage),
            0.0,  # Se calculará después
            json.dumps(decision.suggested_commands)
        ))
        
        conn.commit()
        conn.close()
    
    def _send_intelligent_notification(self, decision: AIDecision, metrics: SystemMetrics):
        """Enviar notificación inteligente con contexto"""
        if not self.config['slack_webhook']:
            return
        
        # Crear mensaje inteligente
        color = {
            'LOW': '#36a64f',      # Verde
            'HIGH': '#ff9500',     # Naranja
            'CRITICAL': '#ff0000'  # Rojo
        }.get(decision.priority, '#36a64f')
        
        emoji = {
            'LOW': '✅',
            'HIGH': '⚠️',
            'CRITICAL': '🚨'
        }.get(decision.priority, '📊')
        
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} Reporte IA del Sistema - {platform.node()}",
                    "fields": [
                        {
                            "title": "Estado",
                            "value": f"CPU: {metrics.cpu_percent}% | RAM: {metrics.memory_percent}%",
                            "short": True
                        },
                        {
                            "title": "Prioridad",
                            "value": decision.priority,
                            "short": True
                        },
                        {
                            "title": "Análisis IA",
                            "value": decision.reasoning,
                            "short": False
                        },
                        {
                            "title": "Acción Recomendada",
                            "value": decision.action,
                            "short": True
                        },
                        {
                            "title": "Confianza",
                            "value": f"{decision.confidence:.1%}",
                            "short": True
                        }
                    ],
                    "timestamp": int(datetime.now().timestamp())
                }
            ]
        }
        
        try:
            response = requests.post(self.config['slack_webhook'], json=message, timeout=10)
            if response.status_code == 200:
                logger.info("📱 Notificación enviada exitosamente")
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
    
    def _execute_safe_commands(self, commands: List[str]):
        """Ejecutar comandos seguros del sistema"""
        import subprocess
        
        safe_commands = [
            'ps', 'top', 'free', 'df', 'du', 'systemctl status',
            'dmesg', 'journalctl', 'uptime', 'whoami'
        ]
        
        for cmd in commands:
            # Verificar que el comando sea seguro
            if any(safe_cmd in cmd for safe_cmd in safe_commands):
                try:
                    logger.info(f"🔧 Ejecutando: {cmd}")
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
                    logger.info(f"✅ Resultado: {result.stdout[:200]}...")
                except Exception as e:
                    logger.error(f"❌ Error ejecutando {cmd}: {e}")
    
    def learn_from_feedback(self, decision_id: int, outcome: str):
        """Aprender de retroalimentación (para mejorar futuras decisiones)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE decisions SET outcome = ? WHERE id = ?',
            (outcome, decision_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"📚 Retroalimentación registrada: {outcome}")
    
    def run_ai_monitoring_cycle(self):
        """Ejecutar ciclo completo de monitoreo con IA"""
        try:
            # 1. Recopilar métricas
            metrics = self.collect_metrics()
            
            # 2. Detectar anomalías
            anomaly_score = self.detect_anomalies(metrics)
            
            # 3. Análisis inteligente
            decision = self.analyze_situation_with_ai(metrics, anomaly_score)
            
            # 4. Ejecutar decisiones
            self.execute_ai_decision(decision, metrics)
            
            # 5. Entrenar modelo si es necesario
            if not self.is_trained and len(self.metrics_history) > 50:
                self._retrain_model()
            
            return metrics, decision
            
        except Exception as e:
            logger.error(f"Error en ciclo de IA: {e}")
            return None, None
    
    def _retrain_model(self):
        """Re-entrenar modelo de anomalías"""
        self._load_historical_data()
        logger.info("🧠 Modelo de anomalías re-entrenado")
    
    def start_ai_monitoring(self):
        """Iniciar monitoreo continuo con IA"""
        logger.info("🚀 Iniciando monitoreo inteligente...")
        
        while True:
            try:
                metrics, decision = self.run_ai_monitoring_cycle()
                
                if metrics and decision:
                    logger.info(f"📊 CPU: {metrics.cpu_percent}% | RAM: {metrics.memory_percent}% | Acción: {decision.action}")
                
                time.sleep(60)  # Esperar 1 minuto
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoreo detenido por el usuario")
                break
            except Exception as e:
                logger.error(f"Error en monitoreo: {e}")
                time.sleep(30)

# Ejemplo de uso
if __name__ == "__main__":
    # Crear agente inteligente
    agent = IntelligentSystemAgent()
    
    print("🤖 Agente de IA para Sistemas")
    print("============================")
    
    # Ejecutar un ciclo de prueba
    print("\n🔍 Ejecutando análisis inteligente...")
    metrics, decision = agent.run_ai_monitoring_cycle()
    
    if metrics and decision:
        print(f"\n📊 Estado del Sistema:")
        print(f"   CPU: {metrics.cpu_percent}%")
        print(f"   RAM: {metrics.memory_percent}%")
        print(f"   Procesos: {metrics.running_processes}")
        
        print(f"\n🤖 Decisión de la IA:")
        print(f"   Acción: {decision.action}")
        print(f"   Prioridad: {decision.priority}")
        print(f"   Razonamiento: {decision.reasoning}")
        print(f"   Confianza: {decision.confidence:.1%}")
        
        if decision.suggested_commands:
            print(f"\n💡 Comandos sugeridos:")
            for i, cmd in enumerate(decision.suggested_commands[:3], 1):
                print(f"   {i}. {cmd}")
    
    print("\n🚀 Para monitoreo continuo, descomenta la última línea del código")
    # agent.start_ai_monitoring()