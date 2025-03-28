import jax.numpy as jnp

A1 = jnp.array(((25, 23.75), (23.75, 25)))
A2 = jnp.array(((25, 23.75), (23.75, 25)))
B1 = jnp.array((4, 6))
B2 = jnp.array((5, 7))
print(jnp.linalg.inv(A1) @ B1)
print(jnp.linalg.inv(A2) @ B2)

A = jnp.array(((25, 0, 23.75, 0), (0, 25, 0, 23.75), (23.75, 0, 25, 0), (0, 23.75, 0, 25)))
B = jnp.array((4, 5, 6, 7))
print(jnp.linalg.inv(A) @ B)