#To define the protocol standards for agents response handling both the input and output formats

from typing import Protocol, List, Dict, Any

class AgentProtocol(Protocol):
    def get_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...