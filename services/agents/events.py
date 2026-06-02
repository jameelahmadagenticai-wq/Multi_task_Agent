from datetime import datetime


class AgentEventManager:

    def __init__(self):

        self.events = []

    def emit(
        self,
        event_type: str,
        agent: str,
        message: str
    ):

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "agent": agent,
            "message": message
        }

        self.events.append(event)

        print(f"\nEVENT: {event}")


event_manager = AgentEventManager()