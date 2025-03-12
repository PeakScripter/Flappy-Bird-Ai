import pygame
import random
import sys
import os
import neat
import numpy as np
import pickle
from datetime import datetime

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
FLAP_STRENGTH = -6
PIPE_SPEED = 3
PIPE_GAP = 150
PIPE_FREQUENCY = 1600  # milliseconds
BIRD_X = 100

# NEAT constants
MAX_FITNESS = 100000  # Limit to prevent overflow

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
SKY_BLUE = (135, 206, 235)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Ai - Automation')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)

def load_images():
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created assets directory. Please add bird and pipe images to this folder.")
        print("The game will run with colored shapes until images are provided.")
        return None, None, None
    
    try:
        bird_img = pygame.image.load('assets/bird.png').convert_alpha()
        bird_img = pygame.transform.scale(bird_img, (30, 30))
        
        pipe_img = pygame.image.load('assets/pilar.png').convert_alpha()
        pipe_img = pygame.transform.scale(pipe_img, (50, 400))
        
        bg_img = pygame.image.load('assets/sky.jpeg').convert()
        bg_img = pygame.transform.scale(bg_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        
        return bird_img, pipe_img, bg_img
    except pygame.error:
        print("Could not load images. Using colored shapes instead.")
        return None, None, None

bird_img, pipe_img, bg_img = load_images()

generation = 0
best_score = 0
best_fitness = 0

class Bird:
    def __init__(self):
        self.x = BIRD_X
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.radius = 15
        self.alive = True
        self.score = 0
        self.fitness = 0
        self.lifetime = 0  
        self.distance_traveled = 0  
        self.angle = 0  
        
    def flap(self):
        if self.alive:
            self.velocity = FLAP_STRENGTH
            self.angle = -30  
        
    def update(self):
        if not self.alive:
            return
            
        self.velocity += GRAVITY
        self.y += self.velocity
        
        if self.velocity < 0:
            self.angle = max(-30, self.angle - 2)  
        else:
            self.angle = min(45, self.angle + 2)  
        
        if self.y < 0:
            self.y = 0
            self.velocity = 0
        
        # Check for floor collision
        if self.y > SCREEN_HEIGHT - 20:
            self.alive = False
        
        # Increment lifetime and distance
        self.lifetime += 1
        self.distance_traveled += PIPE_SPEED
        
    def draw(self, color=YELLOW):
        if bird_img:
            # Rotate the bird image based on velocity
            rotated_bird = pygame.transform.rotate(bird_img, self.angle)
            rect = rotated_bird.get_rect(center=(int(self.x), int(self.y)))
            screen.blit(rotated_bird, rect)
        else:
            # Fallback to circle if no image
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        
    def get_rect(self):
        if bird_img:
            return pygame.Rect(self.x - 15, self.y - 15, 30, 30)
        else:
            return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                              self.radius * 2, self.radius * 2)
        
class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(150, 350)  # More balanced heights
        self.top_rect = pygame.Rect(self.x, 0, 50, self.height)
        self.bottom_rect = pygame.Rect(self.x, self.height + PIPE_GAP, 50, SCREEN_HEIGHT)
        self.passed = False
        
    def update(self):
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x
        
    def draw(self):
        if pipe_img:
            # Draw top pipe (flipped)
            flipped_pipe = pygame.transform.flip(pipe_img, False, True)
            screen.blit(flipped_pipe, (self.x, self.height - 400))  # Adjust position to match hitbox
            
            # Draw bottom pipe
            screen.blit(pipe_img, (self.x, self.height + PIPE_GAP))
        else:
            # Fallback to rectangles if no image
            pygame.draw.rect(screen, GREEN, self.top_rect)
            pygame.draw.rect(screen, GREEN, self.bottom_rect)
        
    def off_screen(self):
        return self.x < -50
    
    def get_gap_center(self):
        return self.height + PIPE_GAP // 2

def calculate_distance_to_gap(bird, pipe):
    # Calculate horizontal distance to pipe
    horizontal_distance = pipe.x - bird.x
    
    # Calculate vertical distance to the center of the gap
    gap_center = pipe.height + PIPE_GAP/2
    vertical_distance = bird.y - gap_center
    
    # Return Euclidean distance to the center of the gap
    return (horizontal_distance**2 + vertical_distance**2)**0.5

def save_best_network(genome, config, score):
    # Create directory if it doesn't exist
    if not os.path.exists('best_birds'):
        os.makedirs('best_birds')
        
    # Save the genome with timestamp and score
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"best_birds/flappy_bird_genome_s{score}_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(genome, f)
    
    print(f"Saved best genome with score {score} to {filename}")

def eval_genomes(genomes, config):
    global generation, best_score, best_fitness
    
    # Track networks and birds for each genome
    nets = []
    birds = []
    ge = []
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome.fitness = 0
        ge.append(genome)
    
    pipes = []
    max_score = 0  # Track the max score in this generation
    
    # Create initial pipe
    pipes.append(Pipe())
    
    # Set up pipe spawning timer
    last_pipe_time = pygame.time.get_ticks()
    
    run = True
    while run and len(birds) > 0:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    run = False
                if event.key == pygame.K_s and max_score > best_score:
                    # Save the best genome in this generation
                    best_idx = birds.index(max(birds, key=lambda x: x.score))
                    save_best_network(ge[best_idx], config, birds[best_idx].score)
        
        # Spawn pipes with consistent timing
        current_time = pygame.time.get_ticks()
        if current_time - last_pipe_time > PIPE_FREQUENCY:
            pipes.append(Pipe())
            last_pipe_time = current_time
        
        # Update birds and get neural network decisions
        for i, bird in enumerate(birds):
            if not bird.alive:
                continue
                
            bird.update()
            
            # Add base fitness for surviving (small increment)
            ge[i].fitness += 0.1
            bird.fitness += 0.1
            
            # Find the next pipe (the one the bird hasn't passed yet)
            next_pipe = None
            for pipe in pipes:
                if not pipe.passed or pipe.x + 50 > bird.x:
                    next_pipe = pipe
                    break
                    
            if next_pipe is None:
                continue
                
            # Calculate distance to the center of the pipe gap
            dist_to_gap = calculate_distance_to_gap(bird, next_pipe)
            
            # Add fitness based on closeness to the center of the gap
            gap_center = next_pipe.height + PIPE_GAP/2
            vertical_dist_to_center = abs(bird.y - gap_center)
            
            # Reward being close to the center of the gap (only when approaching the pipe)
            if next_pipe.x - bird.x < 150 and next_pipe.x > bird.x:
                closeness_reward = 0.01 * (1 - (vertical_dist_to_center / (SCREEN_HEIGHT/2)))
                ge[i].fitness += closeness_reward
                bird.fitness += closeness_reward
            
            # Get neural network inputs
            inputs = [
                bird.y / SCREEN_HEIGHT,  # Bird's height (normalized)
                bird.velocity / 10,  # Bird's velocity (normalized)
                next_pipe.x / SCREEN_WIDTH,  # Distance to pipe (normalized)
                next_pipe.height / SCREEN_HEIGHT,  # Top pipe height (normalized)
                (next_pipe.height + PIPE_GAP) / SCREEN_HEIGHT,  # Bottom pipe y-position (normalized)
                dist_to_gap / SCREEN_WIDTH  # Distance to center of gap (normalized)
            ]
            
            # Add additional inputs for the second pipe if available
            second_pipe = None
            if len(pipes) > 1 and pipes.index(next_pipe) + 1 < len(pipes):
                second_pipe = pipes[pipes.index(next_pipe) + 1]
                inputs.extend([
                    second_pipe.x / SCREEN_WIDTH,  # Second pipe x position
                    second_pipe.height / SCREEN_HEIGHT,  # Second pipe top height
                    (second_pipe.height + PIPE_GAP) / SCREEN_HEIGHT  # Second pipe bottom y-position
                ])
            else:
                # Add placeholder values if no second pipe
                inputs.extend([1.0, 0.5, 0.5])
            
            # Get network output
            output = nets[i].activate(inputs)
            
            # Flap if output exceeds threshold
            if output[0] > 0.5:
                bird.flap()
        
        # Update pipes and check for scoring/collisions
        rem_pipes = []
        for pipe in pipes:
            pipe.update()
            
            # Check for collisions and scoring
            for i, bird in enumerate(birds):
                if not bird.alive:
                    continue
                    
                if not pipe.passed and pipe.x + 50 < bird.x:
                    pipe.passed = True
                    
                    bird.score += 1
                    if bird.score > max_score:
                        max_score = bird.score
                    
                    passing_reward = 5.0
                    ge[i].fitness += passing_reward
                    bird.fitness += passing_reward
                    
                    gap_center = pipe.height + PIPE_GAP/2
                    center_accuracy = 1.0 - min(abs(bird.y - gap_center) / (PIPE_GAP/2), 1.0)
                    center_reward = 2.0 * center_accuracy
                    
                    ge[i].fitness += center_reward
                    bird.fitness += center_reward
                    
                    if bird.score > best_score:
                        best_score = bird.score
                        best_fitness = max(best_fitness, bird.fitness)
                        if bird.score >= best_score + 5:
                            save_best_network(ge[i], config, bird.score)
                
                if (pipe.top_rect.colliderect(bird.get_rect()) or 
                    pipe.bottom_rect.colliderect(bird.get_rect())):
                    bird.alive = False
            
            if pipe.off_screen():
                rem_pipes.append(pipe)
        
        for pipe in rem_pipes:
            pipes.remove(pipe)
        
        birds_to_remove = []
        for i, bird in enumerate(birds):
            if not bird.alive:
                birds_to_remove.append(i)
        
        for i in sorted(birds_to_remove, reverse=True):
            birds.pop(i)
            nets.pop(i)
            ge.pop(i)
        
        if bg_img:
            screen.blit(bg_img, (0, 0))
        else:
            screen.fill(SKY_BLUE)
        
        for pipe in pipes:
            pipe.draw()
        
        # Draw birds with different colors based on score
        for i, bird in enumerate(birds):
            # Color based on score (better birds are more blue)
            if bird.score >= best_score and best_score > 0:
                bird.draw(BLUE)  # Best birds are blue
            elif bird.score > max_score / 2:
                bird.draw(GREEN)  # Good birds are green
            elif bird.score > 0:
                bird.draw(YELLOW)  # Average birds are yellow
            else:
                bird.draw(RED)  # New birds are red
        
        pygame.draw.rect(screen, GREEN, (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20))
        
        gen_text = font.render(f'Generation: {generation}', True, WHITE)
        alive_text = font.render(f'Birds Alive: {len(birds)}', True, WHITE)
        score_text = font.render(f'Current Max Score: {max_score}', True, WHITE)
        best_text = font.render(f'Best Score Ever: {best_score}', True, WHITE)
        
        screen.blit(gen_text, (10, 10))
        screen.blit(alive_text, (10, 30))
        screen.blit(score_text, (10, 50))
        screen.blit(best_text, (10, 70))
        
        pygame.display.update()
        clock.tick(60)
    
    generation += 1

def load_and_run_best_network(config_path, genome_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    bird = Bird()
    pipes = []
    score = 0
    
    pipes.append(Pipe())
    
    last_pipe_time = pygame.time.get_ticks()
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    run = False
        
        if not bird.alive:
            run = False
            break
        
        current_time = pygame.time.get_ticks()
        if current_time - last_pipe_time > PIPE_FREQUENCY:
            pipes.append(Pipe())
            last_pipe_time = current_time
        
        next_pipe = None
        for pipe in pipes:
            if not pipe.passed or pipe.x + 50 > bird.x:
                next_pipe = pipe
                break
                
        if next_pipe is not None:
            dist_to_gap = calculate_distance_to_gap(bird, next_pipe)
            
            inputs = [
                bird.y / SCREEN_HEIGHT, 
                bird.velocity / 10,  
                next_pipe.x / SCREEN_WIDTH, 
                next_pipe.height / SCREEN_HEIGHT, 
                (next_pipe.height + PIPE_GAP) / SCREEN_HEIGHT,  
                dist_to_gap / SCREEN_WIDTH  
            ]
            
            second_pipe = None
            if len(pipes) > 1 and pipes.index(next_pipe) + 1 < len(pipes):
                second_pipe = pipes[pipes.index(next_pipe) + 1]
                inputs.extend([
                    second_pipe.x / SCREEN_WIDTH,  
                    second_pipe.height / SCREEN_HEIGHT, 
                    (second_pipe.height + PIPE_GAP) / SCREEN_HEIGHT 
                ])
            else:
                inputs.extend([1.0, 0.5, 0.5])
            
            output = net.activate(inputs)
            
            if output[0] > 0.5:
                bird.flap()
        
        bird.update()
        
        rem_pipes = []
        for pipe in pipes:
            pipe.update()
            
            if not pipe.passed and pipe.x + 50 < bird.x:
                pipe.passed = True
                score += 1
            
            if pipe.top_rect.colliderect(bird.get_rect()) or pipe.bottom_rect.colliderect(bird.get_rect()):
                bird.alive = False
            
            if pipe.off_screen():
                rem_pipes.append(pipe)
        
        for pipe in rem_pipes:
            pipes.remove(pipe)
        
        if bg_img:
            screen.blit(bg_img, (0, 0))
        else:
            screen.fill(SKY_BLUE)
        
        for pipe in pipes:
            pipe.draw()
        
        bird.draw(BLUE)
        
        pygame.draw.rect(screen, GREEN, (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20))
        
        score_text = font.render(f'Score: {score}', True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.update()
        clock.tick(60)
    
    print(f"Final Score: {score}")
    return score

def run_neat(config_path):
    global generation
    generation = 0
    
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    population = neat.Population(config)
    
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    if not os.path.exists('best_birds'):
        os.makedirs('best_birds')
    
    winner = population.run(eval_genomes, 100)  
    
    with open('best_birds/winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    print('\nBest genome:\n{!s}'.format(winner))
    
    print("\nRunning the best genome...")
    load_and_run_best_network(config_path, 'best_birds/winner.pkl')

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        if len(sys.argv) > 2:
            # config_path = create_config_file()
            genome_path = sys.argv[2]
            if os.path.exists(genome_path):
                load_and_run_best_network("./config.txt", genome_path)
            else:
                print(f"Genome file not found: {genome_path}")
        else:
            # config_path = create_config_file()
            if os.path.exists('best_birds/winner.pkl'):
                load_and_run_best_network("./config.txt", 'best_birds/winner.pkl')
            else:
                print("No winner.pkl found. Please train the network first.")
    else:
        # config_path = create_config_file()
        
        run_neat("./config.txt")