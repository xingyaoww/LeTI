TEMPLATE_CODE = """
from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class FAC(Entity):
    \"\"\"A functional, primarily man-made structure. Facilities are artifacts falling under the domains of architecture and civil engineering, including more temporary human constructs, such as police lines and checkpoints.\"\"\"
    def __init__(self, name: str):
        super().__init__(name=name)

class GPE(Entity):
    \"\"\"Geopolitical entities such as countries, provinces, states, cities, towns, etc. GPEs are composite entities, consisting of a physical location, a government, and a population. All three of these elements must be present for an entity to be tagged as a GPE. A GPE entity may be a single geopolitical entity or a group.\"\"\"
    def __init__(self, name: str):
        super().__init__(name=name)

class LOC(Entity):
    \"\"\"Geographical entities such as geographical areas and landmasses, bodies of water\"\"\"
    def __init__(self, name: str):
        super().__init__(name=name)

class ORG(Entity):
    \"\"\"Corporations, agencies, and other groups of people defined by an established organizational structure. An ORG entity may be a single organization or a group. A key feature of an ORG is that it can change members without changing identity.\"\"\"
    def __init__(self, name: str):
        super().__init__(name=name)

class PER(Entity):
    \"\"\"Person entities are limited to humans. A PER entity may be a single person or a group.\"\"\"
    def __init__(self, name: str):
        super().__init__(name=name)

class VEH(Entity):
    \"\"\"A physical device primarily designed to move an object from one location to another, by (for example) carrying, flying, pulling, or pushing the transported object. Vehicle entities may or may not have their own power source.\"\"\"
    def __init__(self, name: str):
        super().__init__(name=name)

class WEA(Entity):
    \"\"\"A physical device that is primarily used as an instrument for physically harming or destroying entities\"\"\"
    def __init__(self, name: str):
        super().__init__(name=name)

class Movement(Event):
    def __init__(
        self,
        agent: List[GPE | ORG | PER] = [],
        artifact: List[FAC | ORG | PER | VEH | WEA] = [],
        destination: List[FAC | GPE | LOC] = [],
        origin: List[FAC | GPE | LOC] = [],
        vehicle: List[VEH] = [],
    ):
        self.agent = agent
        self.artifact = artifact
        self.destination = destination
        self.origin = origin
        self.vehicle = vehicle


class Transport(Movement):
    \"\"\"self.agent transported self.artifact in self.vehicle vehicle from self.origin place to self.destination place.\"\"\"
    def __init__(
        self,
        agent: List[GPE | ORG | PER] = [],
        artifact: List[FAC | ORG | PER | VEH | WEA] = [],
        destination: List[FAC | GPE | LOC] = [],
        origin: List[FAC | GPE | LOC] = [],
        vehicle: List[VEH] = [],
    ):
        super().__init__(
            agent=agent,
            artifact=artifact,
            destination=destination,
            origin=origin,
            vehicle=vehicle,
        )

\"\"\"
Translate the following sentence into an instance of Transport. The trigger word(s) of the event is marked with **trigger word**.
"Even as the secretary of homeland security was putting his people on high alert last month , a 30-foot Cuban patrol boat with four heavily armed men **landed** on American shores , utterly undetected by the Coast Guard Secretary Ridge now leads ."
\"\"\"
{}
"""

REFERENCE = """
import sys
sys.path.insert(0, "/root/lang-reward")
from leti.verifier.eae import EAESolutionVerifier

verifier = EAESolutionVerifier(
    gold_set={(21, 21, 'Movement:Transport', 'Vehicle', 'VEH'), (30, 30, 'Movement:Transport', 'Destination', 'LOC'), (26, 26, 'Movement:Transport', 'Artifact', 'PER')},
    cur_event={'event_type': 'Movement:Transport', 'id': 'CNN_CF_20030303.1900.00-6-EV0', 'trigger': {'start': 27, 'end': 28, 'text': 'landed'}, 'arguments': [{'entity_id': 'CNN_CF_20030303.1900.00-6-E5', 'text': 'boat', 'role': 'Vehicle'}, {'entity_id': 'CNN_CF_20030303.1900.00-6-E6', 'text': 'men', 'role': 'Artifact'}, {'entity_id': 'CNN_CF_20030303.1900.00-6-E8', 'text': 'shores', 'role': 'Destination'}]},
    tokens=['Even', 'as', 'the', 'secretary', 'of', 'homeland', 'security', 'was', 'putting', 'his', 'people', 'on', 'high', 'alert', 'last', 'month', ',', 'a', '30-foot', 'Cuban', 'patrol', 'boat', 'with', 'four', 'heavily', 'armed', 'men', 'landed', 'on', 'American', 'shores', ',', 'utterly', 'undetected', 'by', 'the', 'Coast', 'Guard', 'Secretary', 'Ridge', 'now', 'leads', '.'],
)
verifier.verify(transport_event)
"""


from leti.utils.execute import unsafe_execute_mp

CODE = TEMPLATE_CODE.format(
"""transport_event = Transport(
    vehicle=[
        VEH("boat"),
    ],
    artifact=[
        PER("men"),
    ],
    destination=[
        LOC("shores"),
    ],
)
""") + REFERENCE

result = unsafe_execute_mp(
    CODE,
    extra_headers="from __future__ import annotations",
    timeout=10
)
import json
print(json.dumps(result, indent=4))

BUGGY1 = TEMPLATE_CODE.format(
"""transport_event = Transport(
    vehicle=[
        VEH,
    ],
    artifact=[
        PER,
    ],
    destination=[
        LOC,
    ],
)
""") + REFERENCE
result = unsafe_execute_mp(
    BUGGY1,
    extra_headers="from __future__ import annotations",
    timeout=10
)
print(json.dumps(result, indent=4))


BUGGY2 = TEMPLATE_CODE.format(
"""transport_event = Transport(
    vehicle=[
        VEH("boat"),
    ],
    artifact=[
        PER("men"),
    ],
)
""") + REFERENCE
result = unsafe_execute_mp(
    BUGGY2,
    extra_headers="from __future__ import annotations",
    timeout=10
)
print(json.dumps(result, indent=4))


BUGGY2 = TEMPLATE_CODE.format(
"""transport_event = Transport(
    vehicle=[
        VEH("men"),
    ],
    artifact=[
        PER("boat"),
    ],
)
""") + REFERENCE
result = unsafe_execute_mp(
    BUGGY2,
    extra_headers="from __future__ import annotations",
    timeout=10
)
print(json.dumps(result, indent=4))


BUGGY3 = TEMPLATE_CODE.format(
"""transport_event = Transport(
    vehicle=[
        VEH("boat"),
    ],
    artifact=[
        PER("men"),
        LOC("shores"),
    ],
)
""") + REFERENCE

result = unsafe_execute_mp(
    BUGGY3,
    extra_headers="from __future__ import annotations",
    timeout=10
)
print(json.dumps(result, indent=4))


BUGGY4 = TEMPLATE_CODE.format(
"""transport_event = Transport(
    vehicle=None
)
""") + REFERENCE

result = unsafe_execute_mp(
    BUGGY4,
    extra_headers="from __future__ import annotations",
    timeout=10
)
print(json.dumps(result, indent=4))
