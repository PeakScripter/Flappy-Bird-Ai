# Flappy Bird AI with NEAT

![Flappy Bird AI](

https://github.com/user-attachments/assets/af8ef837-514c-4155-9968-8a0f4b725297

)

A Flappy Bird clone that uses the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to train AI agents to play the game. Watch as neural networks evolve from random movements to mastering the game through generations of training.

## Features

- **NEAT Implementation**: Neural networks evolve over generations to improve gameplay
- **Visual Feedback**: Birds are color-coded based on performance (red → yellow → green → blue)
- **Save & Load Models**: Best-performing networks are automatically saved and can be loaded later
- **Performance Tracking**: Monitor generation count, birds alive, current score, and best score
- **Customizable Parameters**: Easily adjust game physics, neural network inputs, and NEAT settings

## Requirements

- Python 3.6+
- pygame
- neat-python
- numpy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/peakscripter/flappy-bird-ai.git
   cd flappy-bird-ai
   ```

2. Install the required packages:
   ```
   pip install pygame neat-python numpy
   ```

3. Create an `assets` folder and add the following images (optional):
   - `bird.png` - The bird sprite
   - `pilar.png` - The pipe sprite
   - `sky.jpeg` - The background image

   If no images are provided, the game will use colored shapes instead.

4. Make sure you have a `config.txt` file in the root directory (provided in the repository).

## Usage

### Training the AI

To start training the neural networks:
```
python final.py
```

During training:
- Press `Q` to quit the current generation
- Press `S` to save the best genome of the current generation

### Running a Trained Model

To run a previously trained model:
```
python final.py --run
```

This will run the best model saved as `best_birds/winner.pkl`.


## How It Works

The NEAT algorithm creates a population of neural networks that control birds in the game. Each network receives inputs about the bird's position, velocity, and the location of upcoming pipes. The network outputs determine whether the bird should flap its wings.

Networks that perform better (survive longer and score more points) have a higher chance of passing their "genes" to the next generation. Over time, the networks evolve to become increasingly skilled at playing the game.

### Neural Network Inputs:
- Bird's height (normalized)
- Bird's velocity (normalized)
- Distance to next pipe (normalized)
- Top pipe height (normalized)
- Bottom pipe y-position (normalized)
- Distance to center of gap (normalized)
- Information about the second upcoming pipe (if available)

## Project Structure

- `final.py` - Main game file containing the NEAT implementation
- `config.txt` - Configuration file for the NEAT algorithm
- `assets/` - Directory for game images
- `best_birds/` - Directory where trained models are saved

## License

[MIT License](LICENSE)

## Acknowledgments

- [NEAT-Python](https://neat-python.readthedocs.io/) for the implementation of the NEAT algorithm
- [Pygame](https://www.pygame.org/) for the game framework
