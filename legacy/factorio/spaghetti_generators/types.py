from typing import Literal, Optional

ForkType = Literal["in", "out"]
Fork = tuple[ForkType, str]
Forks = list[Optional[Fork]]
