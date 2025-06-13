"""[Patch v6.9.17] Persistent system state manager."""

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class StateManager:
    """จัดการสถานะของระบบที่ต้องคงอยู่ข้ามการรัน."""

    def __init__(self, state_file_path: str = 'output/system_state.json'):
        self.state_file_path = state_file_path
        self.state: Dict[str, Any] = self._get_default_state()
        self.load_state()

    def _get_default_state(self) -> Dict[str, Any]:
        """คืนค่า state เริ่มต้นทั้งหมด"""
        return {
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'cooldown_status': {},
            'last_trade_pnl': 0.0,
            'active_kill_switch': False,
        }

    def load_state(self) -> None:
        """โหลดค่า state จากไฟล์ JSON"""
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r') as f:
                    loaded_state = json.load(f)
                self.state.update(loaded_state)
                logger.info("Loaded system state from %s", self.state_file_path)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(
                    "Could not read state file %s. Using default state. Error: %s",
                    self.state_file_path,
                    e,
                )
                self.state = self._get_default_state()
        else:
            logger.info("State file not found. Starting with default state.")

    def save_state(self) -> None:
        """บันทึก state ปัจจุบันลงไฟล์ JSON"""
        try:
            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            with open(self.state_file_path, 'w') as f:
                json.dump(self.state, f, indent=4)
            logger.info("System state saved to %s", self.state_file_path)
        except IOError as e:
            logger.error("Could not save state file to %s. Error: %s", self.state_file_path, e)

    def update_state(self, key: str, value: Any) -> bool:
        """อัปเดตค่าใน state และบอกว่ามีการเปลี่ยนแปลงหรือไม่"""
        if key in self.state and self.state[key] != value:
            self.state[key] = value
            return True
        if key not in self.state:
            self.state[key] = value
            return True
        return False

    def get_state(self, key: str, default: Any = None) -> Any:
        """ดึงค่าจาก state ถ้าไม่มีจะคืนค่าตาม default"""
        return self.state.get(key, default)
