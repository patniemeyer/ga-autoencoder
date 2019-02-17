from util import *
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from scipy import stats

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

# np.random.seed(42)
if not os.path.exists('./out/'): os.mkdir('./out/')
# os.system('open ./out');  # exit()
f = open('loss.txt', 'w'); f.truncate(); f.close()

def noise_like(weights: torch.Tensor, seed: int)->torch.Tensor:
    with torch.random.fork_rng():
        torch.random.manual_seed(seed)
        return torch.randn_like(weights)

def apply_noise(weights:torch.Tensor, seed: int, noise_std)->torch.Tensor:
    return weights + noise_like(weights, seed) * noise_std

def fitness_batch(current_weights, seeds: [float], noise_std)->[float]:
    return [fitness(apply_noise(current_weights, seed, noise_std)) for seed in seeds]

def fitness(weights, save=False):
    model.setWeightsFlat(weights)
    epochloss = 0
    for batch in traindataloader:
        img, _ = batch  # [128, 1, 28, 28]
        img = img.view(img.size(0), -1)  # flat [128, 784]
        if torch.cuda.is_available(): img = img.cuda()
        output = model(img)  # [128, 784]

        batch_loss = criterion(output, img).item()  # scalar
        epochloss += batch_loss

    if save:
        images=torch.cat(( to_img(img[0:10]), to_img(output[0:10]) )) # TODO: interleave these
        save_image(images, './out/image_{}.png'.format(epoch), nrow=10)

    return epochloss


epoch = 0
def train(threads=0, num_epochs=100000, target_fitness=0.001,
        pop_size = 1000, # [25]
        noise_std = 1e-3,  # [1e-3]
        learning_rate = 0.001, # [Adam: 0.005, 1e-4, SGD: 0.02]
        weight_decay = 0 # [Adam: 0 or 1e-3]
    ):
    global epoch
    if threads > 0: pool = ProcessPoolExecutor(max_workers=threads)
    weight_current = copy.deepcopy(weight_init)

    opt = torch.optim.Adam(params=[weight_current], lr=learning_rate, weight_decay=weight_decay)
    # opt = torch.optim.SGD(params=[weight_current], lr=learning_rate, weight_decay=weight_decay)

    initial_fitness = fitness(weight_init)
    past_fitnesses = deque(maxlen=10); past_fitnesses.append(initial_fitness)

    for epoch in range(num_epochs):
        start_time = millis()
        seeds = [random.randint(0, 2**32) for _ in range(pop_size)]

        if threads == 0:
            R = [fitness(apply_noise(weight_current, seed, noise_std)) for seed in seeds]
        else:
            seed_batches = list(chunks(seeds, int(len(seeds)/threads)))
            fitnesses = list(pool.map( fitness_batch, [weight_current]*len(seed_batches), seed_batches, [noise_std]*len(seed_batches)))
            R = np.concatenate(fitnesses)

        # standardize the rewards to have a gaussian distribution
        F = (R - np.mean(R)) / (np.std(R)+1e-9)

        # w = w + alpha / (npop * sigma) * np.dot(N.T, A)
        gradient = torch.zeros_like(weight_current)
        for pi in range(pop_size):
            N = noise_like(weight_current, seeds[pi]) # reproduce N
            gradient += F[pi] * N / pop_size

        # gradient = opt.update(gradient)
        # weight_current -= gradient * learning_rate
        # weight_current -= weight_current * decay

        weight_current.grad = gradient
        opt.step() # update the weights with lr and decay

        end_time = millis()
        current_fitness = fitness(weight_current)
        past_fitnesses.append(current_fitness)
        if current_fitness <= target_fitness: break

        f=open('loss.txt', 'a'); print(current_fitness, file=f); f.close()
        if epoch > 0 and epoch % 1 == 0:
            slope, _, _, _, _ = stats.linregress(x=np.arange(len(past_fitnesses)), y=past_fitnesses)
            print('epoch %d. fitness: %f, slope: %f, rate: %f' % (epoch, current_fitness, slope, 1000/(end_time-start_time)))
        if epoch > 0 and epoch % 100 == 0:
            fitness(weight_current, save=True)
            if current_fitness < initial_fitness:
                model.save(); print("saved fitness: ", current_fitness)

traindataloader = get_data(128)
# traindataloader = get_data(2)

# criterion = nn.MSELoss(reduction='sum')
criterion = nn.MSELoss()

# model = autoencoder(filename=None)
model = autoencoder()

if torch.cuda.is_available(): model.cuda()
if model.load():
    weight_init = model.getWeightsFlat()
    print("initial fitness: ", fitness(weight_init))
else:
    weight_init = torch.zeros_like(model.getWeightsFlat())

train(
    threads=8,
    target_fitness=0.01,
    pop_size = 200, # [25]
    noise_std = 1e-3,  # [1e-3]
    learning_rate = 0.005, # [Adam: 0.005, 1e-4, SGD: 0.02]
    weight_decay = 0 # [Adam: 0 or 1e-3]
)
# config_string='pop={}_σ={}_α={}'.format(npop, sigma, alpha)

