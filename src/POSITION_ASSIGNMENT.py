import networkx as nx

# Dictionary that maps position names to position IDs and hierarchy number
# Key: position names (all possible in the dataset)
# Value: tuple of (Position ID, hierarchy number)
position_id_mapping = {"head coach": ("HC",1),
                        "interim head coach": ("iHC", 1),
                        "offensive coordinator": ("OC", 2),
                        "passing game coordinator": ("OC", 2),
                        "run game coordinator": ("OC", 2),
                        "running game coordinator": ("OC", 2),
                        "passing game": ("OC", 2),
                        "pass game coordinator": ("OC", 2),
                        "pass offense coordinator": ("OC", 2),
                        "pass coordinator": ("O", 2),
                        "offensive quality control": ("O", 3),
                        "passing coordinator": ("OC", 2),
                        "offensive passing game coordinator": ("OC", 2),
                        "special teams coordinator": ("SC", 2),
                        "defensive coordinator": ("DC", 2),
                        "defensive passing game coordinator": ("DC", 2),
                        "defensive pass game coordinator": ("DC", 2),
                        "pass defense coordinator": ("DC", 2),
                        "coverage coordinator": ("DC", 2),
                        "defensive front coordinator": ("DC", 2),
                        "run defense coordinator": ("DC", 2),
                        "quarterbacks": ("O", 3),
                        "quaterbacks": ("O", 3),
                        "running backs": ("O", 3),
                        "offensive line": ("O", 3),
                        "wide receivers": ("O", 3),
                        "receivers": ("O", 3),
                        "tight ends": ("O", 3),
                        "tights ends": ("O", 3),
                        "pass rush": ("O", 3),
                        "game management": ("O", 3),
                        "run game": ("O", 3),
                        "running game": ("O", 3),
                        "offensive quality control": ("O", 3),
                        "special teams": ("S", 3),
                        "linebackers": ("D", 3),
                        "defensive line": ("D", 3),
                        "secondary": ("D", 3),
                        "defensive backs": ("D", 3),
                        "inside linebackers": ("D", 3),
                        "outside linebackers": ("D", 3),
                        "outisde linebackers": ("D", 3),
                        "cornerbacks": ("D", 3),
                        "safeties": ("D", 3),
                        "nickel backs": ("D", 3),
                        "defensive nickel package": ("D", 3),
                        "defensive ends": ("D", 3),
                        "defensive tackles": ("D", 3),
                        "defensive secondary": ("D", 3),
                        "defensive front seven": ("D", 3),
                        "defensive tackles": ("D", 3),
                        "third down": ("D", 3),
                        "defensive quality control": ("D", 3)}

