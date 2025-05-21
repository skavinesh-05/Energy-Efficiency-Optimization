import random
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from collections import defaultdict

class EnergyEfficiencyOptimizer:
    def __init__(self):
        # Initialize system components
        self.sensors = {}
        self.automation_rules = []
        self.energy_data = pd.DataFrame(columns=['timestamp', 'device', 'energy_usage', 'occupancy', 'temp'])
        self.ai_model = None
        self.user_preferences = {}
        self.blockchain_log = []
        
        # Initialize with some default devices
        self._initialize_default_devices()
        
    def _initialize_default_devices(self):
        """Initialize some default devices for simulation"""
        default_devices = {
            'HVAC': {'type': 'climate', 'current_setting': 22, 'min_temp': 18, 'max_temp': 26},
            'Lighting_LivingRoom': {'type': 'lighting', 'current_state': 'off'},
            'Lighting_Kitchen': {'type': 'lighting', 'current_state': 'off'},
            'Refrigerator': {'type': 'appliance', 'current_state': 'on'},
            'Water_Heater': {'type': 'appliance', 'current_state': 'on'}
        }
        
        for device, params in default_devices.items():
            self.add_sensor(device, params['type'], params)
    
    def add_sensor(self, device_name, device_type, params):
        """Add a new sensor/device to the monitoring system"""
        self.sensors[device_name] = {
            'type': device_type,
            'params': params,
            'history': []
        }
    
    def add_automation_rule(self, condition_func, action_func, description):
        """Add a new automation rule"""
        self.automation_rules.append({
            'condition': condition_func,
            'action': action_func,
            'description': description
        })
    
    def collect_sensor_data(self):
        """Simulate collecting data from all sensors"""
        current_time = datetime.now()
        occupancy = random.choice(['low', 'medium', 'high'])
        temp = random.uniform(15, 30)  # Simulate outdoor temperature
        
        for device, data in self.sensors.items():
            # Simulate energy usage based on device type and state
            if data['type'] == 'climate':
                usage = random.uniform(1.5, 3.0) * abs(data['params']['current_setting'] - temp) / 5
            elif data['type'] == 'lighting':
                usage = 0.1 if data['params']['current_state'] == 'on' else 0
            else:  # appliances
                usage = random.uniform(0.5, 1.5) if data['params']['current_state'] == 'on' else 0
            
            # Store the data
            new_entry = {
                'timestamp': current_time,
                'device': device,
                'energy_usage': round(usage, 2),
                'occupancy': occupancy,
                'temp': round(temp, 1)
            }
            
            self.energy_data = pd.concat([self.energy_data, pd.DataFrame([new_entry])], ignore_index=True)
            data['history'].append(new_entry)
            
            # Log to blockchain
            self._add_to_blockchain(device, new_entry)
        
        return self.energy_data.tail(len(self.sensors))
    
    def _add_to_blockchain(self, device, data):
        """Simulate adding data to blockchain"""
        block = {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'data': data,
            'previous_hash': self.blockchain_log[-1]['hash'] if self.blockchain_log else None,
            'hash': hash(json.dumps(data, default=str))
        }
        self.blockchain_log.append(block)
    
    def apply_automation_rules(self):
        """Apply all automation rules based on current conditions"""
        actions_taken = []
        current_data = self.energy_data.tail(len(self.sensors)).to_dict('records')
        
        for rule in self.automation_rules:
            for data in current_data:
                if rule['condition'](data):
                    action_result = rule['action'](data)
                    actions_taken.append({
                        'rule': rule['description'],
                        'device': data['device'],
                        'action': action_result,
                        'timestamp': datetime.now()
                    })
        
        return actions_taken
    
    def train_ai_model(self):
        """Train the AI model for energy usage prediction"""
        if len(self.energy_data) < 100:
            print("Not enough data to train model. Need at least 100 records.")
            return False
        
        # Prepare data
        df = self.energy_data.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['occupancy_num'] = df['occupancy'].map({'low': 0, 'medium': 1, 'high': 2})
        
        # Features and target
        X = df[['hour', 'occupancy_num', 'temp']]
        y = df['energy_usage']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        self.ai_model = RandomForestRegressor(n_estimators=100)
        self.ai_model.fit(X_train, y_train)
        
        # Evaluate
        score = self.ai_model.score(X_test, y_test)
        print(f"Model trained with R^2 score: {score:.2f}")
        return True
    
    def generate_recommendations(self):
        """Generate energy optimization recommendations using AI"""
        if not self.ai_model:
            print("AI model not trained yet.")
            return []
        
        # Get recent data
        recent_data = self.energy_data.tail(24 * len(self.sensors))  # Last 24 hours
        
        # Predict optimal settings
        recommendations = []
        for device in self.sensors:
            device_data = recent_data[recent_data['device'] == device]
            if len(device_data) == 0:
                continue
            
            # Analyze patterns
            avg_usage = device_data['energy_usage'].mean()
            peak_hours = device_data.groupby(device_data['timestamp'].dt.hour)['energy_usage'].mean()
            
            # Generate recommendation
            rec = {
                'device': device,
                'current_avg_usage': round(avg_usage, 2),
                'peak_hours': peak_hours.idxmax(),
                'suggestions': []
            }
            
            if self.sensors[device]['type'] == 'climate':
                current_temp = self.sensors[device]['params']['current_setting']
                optimal_temp = current_temp + (1 if avg_usage > 2.5 else -1)
                rec['suggestions'].append(f"Adjust temperature from {current_temp}°C to {optimal_temp}°C")
            
            elif self.sensors[device]['type'] == 'lighting':
                if avg_usage > 0.5:
                    rec['suggestions'].append("Consider using motion sensors or timers to reduce lighting usage")
            
            recommendations.append(rec)
        
        return recommendations
    
    def visualize_energy_usage(self, period='day'):
        """Generate visualizations of energy usage"""
        if len(self.energy_data) == 0:
            print("No data to visualize")
            return
        
        df = self.energy_data.copy()
        df['time_group'] = df['timestamp'].dt.floor(period[0])
        
        plt.figure(figsize=(12, 6))
        
        if period == 'day':
            # Daily usage by device
            pivot = df.pivot_table(index='time_group', columns='device', values='energy_usage', aggfunc='sum')
            pivot.plot(kind='bar', stacked=True)
            plt.title("Daily Energy Usage by Device")
            plt.ylabel("Energy Usage (kWh)")
        
        elif period == 'hour':
            # Hourly pattern
            df['hour'] = df['timestamp'].dt.hour
            pivot = df.pivot_table(index='hour', columns='device', values='energy_usage', aggfunc='mean')
            pivot.plot(kind='line')
            plt.title("Average Hourly Energy Usage Pattern")
            plt.ylabel("Average Energy Usage (kWh)")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def simulate_operation(self, days=1):
        """Simulate system operation for a given number of days"""
        results = []
        start_time = datetime.now()
        
        for _ in range(days * 24):  # Simulate each hour
            # Simulate time passing
            current_time = start_time + timedelta(hours=_)
            print(f"\n--- Hour {_+1}: {current_time.strftime('%Y-%m-%d %H:%M')} ---")
            
            # Collect sensor data
            sensor_data = self.collect_sensor_data()
            print("Sensor data collected for:", ", ".join(sensor_data['device'].unique()))
            
            # Apply automation rules
            actions = self.apply_automation_rules()
            for action in actions:
                print(f"Automation: {action['rule']} for {action['device']}")
            
            # Every 6 hours, generate recommendations
            if _ % 6 == 0 and _ > 0:
                if len(self.energy_data) >= 100 and not self.ai_model:
                    self.train_ai_model()
                
                if self.ai_model:
                    recs = self.generate_recommendations()
                    for rec in recs:
                        print(f"\nRecommendation for {rec['device']}:")
                        for sug in rec['suggestions']:
                            print(f"- {sug}")
            
            # Store results
            results.append({
                'hour': _+1,
                'timestamp': current_time,
                'total_energy': sensor_data['energy_usage'].sum(),
                'actions_taken': len(actions)
            })
            
            # Sleep to simulate real-time (remove for faster simulation)
            time.sleep(0.1)
        
        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Initialize the system
    optimizer = EnergyEfficiencyOptimizer()
    
    # Add some automation rules
    optimizer.add_automation_rule(
        condition_func=lambda data: data['occupancy'] == 'low' and data['device'].startswith('Lighting'),
        action_func=lambda data: optimizer.sensors[data['device']]['params'].update({'current_state': 'off'}),
        description="Turn off lights in low occupancy"
    )
    
    optimizer.add_automation_rule(
        condition_func=lambda data: data['temp'] > 25 and data['device'] == 'HVAC',
        action_func=lambda data: optimizer.sensors[data['device']]['params'].update({'current_setting': max(22, optimizer.sensors[data['device']]['params']['current_setting'] - 1)}),
        description="Adjust HVAC for high temperatures"
    )
    
    # Simulate 3 days of operation
    print("Starting energy efficiency optimization simulation...")
    simulation_results = optimizer.simulate_operation(days=3)
    
    # Show results
    print("\nSimulation complete. Showing summary...")
    print(f"Total energy used: {simulation_results['total_energy'].sum():.2f} kWh")
    print(f"Total automation actions taken: {simulation_results['actions_taken'].sum()}")
    
    # Visualize results
    optimizer.visualize_energy_usage(period='day')
    optimizer.visualize_energy_usage(period='hour')
    
    # Show some recommendations
    print("\nFinal recommendations:")
    final_recs = optimizer.generate_recommendations()
    for rec in final_recs:
        print(f"\n{rec['device']} (avg usage: {rec['current_avg_usage']} kWh):")
        for sug in rec['suggestions']:
            print(f"- {sug}")