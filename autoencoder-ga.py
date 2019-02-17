from util import *
from ga_opt import GAOptimizer

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not os.path.exists('./out/'): os.mkdir('./out/')
os.system('open ./out'); #exit()

def fitness(individual: Individual, save=False)->float:
    model.setWeights(individual.genes)
    epochloss=0
    for batch in batches:
        img, _ = batch  # [128, 1, 28, 28]
        img = img.view(img.size(0), -1)  # flat [128, 784]
        if torch.cuda.is_available(): img = img.cuda()
        output = model(img)  # [128, 784]

        batch_loss = criterion(output, img).item() # scalar
        epochloss += batch_loss

    if save:
        epoch = ga.iteration_num
        images=torch.cat(( to_img(img[0:10]), to_img(output[0:10]) )) # interleave these
        save_image(images, './out/image_{}.png'.format(epoch), nrow=10)
        output_diff = nn.MSELoss()(output[0], output[1]).item()
        print("diff = ", output_diff)

    return epochloss

# model = autoencoder(filename=None)
model = autoencoder()
loaded_model = model.load()
if torch.cuda.is_available(): model.cuda()

criterion = nn.MSELoss()
weight_start = list(model.getWeights())

batches = get_data(128)
# batches = get_data(2)

# e.g. images=2, pop=200, mut=1.0, cross=0.0, sigma=0.5, single value mutations gets to 0.1 in 12k iters
# e.g. images=128, pop=1000, mut=1.0, cross=0.0, sigma=1.0, single value mutations gets to 4350 in 15k iters
ga = GeneticAlgorithm(
    population_size=200, # [200]
    crossover_probability=0.1,
    mutation_probability=0.9,
    elitism=True,
    maximise_fitness=False)

sigma = 1.0 # [1.0?] large values seem ok here for single mutations

ga_opt = GAOptimizer(ga, model, loaded_model, weight_start, criterion, batches, fitness, sigma)

ga.create_individual = ga_opt.create_individual
# ga.selection_function = ga.tournament_selection
ga.selection_function = ga.elite_selection
# ga.selection_function = ga.roulette_selection

# ga.crossover_function = two_point_crossover_one_weight
ga.crossover_function = ga_opt.swap_two_layers

# ga.mutate_function = mutate_all_weights
ga.mutate_function = ga_opt.mutate_one_weight

ga.fitness_function = fitness
ga.iteration_function = ga_opt.report

f=open('loss.txt', 'w'); f.truncate(); f.close()

# print(len(list(model.getWeights())))
# for p in model.parameters():
#     print(p.name, p.shape)
# exit()

ga.run(50000)


