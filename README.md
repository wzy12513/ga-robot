# ga-robot

ctrrn use `https://rdbeer.pages.iu.edu/`

GA use `https://github.com/Arash-codedev/openGA`

mujoco use `https://github.com/google-deepmind/mujoco`

## guide

to build it, ensure your linux already install mujoco

here is how to install mujoco

```bash
git clone https://github.com/google-deepmind/mujoco.git
cd mujoco
mkdir build
cd build
cmake ..
make
make --install
```

how to train

```bash
git clone https://github.com/wzy12513/ga-robot.git
cd ga-robot
cd mujoco
mkdir build
cd build
cmake ..
make
cd bin
./train <path to your module>
```

I use `./train ../../model/humanoid/humanoid.xml`

## TODO

- [ ] **train to move**
- [ ] transfer learning