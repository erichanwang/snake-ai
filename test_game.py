from snake_game import snakegame

g=snakegame()
s=g.reset()
print("start:",s)

steps=0
alive=True
while alive and steps<50:
    s,alive,score=g.step(0)#always go up
    steps+=1
    print(f"step {steps}: alive={alive}, score={score}")

print(f"survived {steps} steps")
