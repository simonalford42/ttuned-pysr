from sr import BasicSR
from problems import pythagorean_3d

X, y = pythagorean_3d(seed=42)
model = BasicSR(
                 population_size=20,
                 num_generations=10000,
                 max_depth=4,
                 max_size=15,
                 tournament_size=3,
                 binary_operators=['+', '*'],
                 unary_operators=[],
                 constants=[],
                 collect_trajectory=True,
                 record_heritage=True,
)

model.fit(X, y, verbose=True)
# print out the heritage
heritage = model.retrieve_heritage_of_best_expression()
print([len(g) for g in heritage])
print(heritage)

import pdb; pdb.set_trace()
