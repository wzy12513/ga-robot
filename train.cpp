#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include "openGA.h"
#include "RobotController.h"
#include <mujoco/mujoco.h>

static const int NET_N = 32;          // CTRNN neurons
static const int MOTOR_COUNT = 17;    // humanoid actuators
static const double SIM_TIME = 3.0;   // each evaluation seconds

struct Chromo {
    std::vector<double> p;
    Chromo(){ p.resize(NET_N*NET_N + NET_N + NET_N); }
};

struct Result {
    double fitness;
};

typedef EA::Genetic<Chromo, Result> GA;

double evaluate(const Chromo& C, mjModel* m)
{
    mjData* d = mj_makeData(m);
    mj_resetData(m, d);
    mj_forward(m, d);

    RobotController ctrl(NET_N, MOTOR_COUNT, m->opt.timestep);
    ctrl.loadParameters(C.p);
    ctrl.reset();

    double x0 = d->qpos[0];

    int total_steps = SIM_TIME / m->opt.timestep;

    for(int t=0;t<total_steps;t++)
    {
        std::vector<double> sensors;
        for(int i=0;i<m->nq;i++) sensors.push_back(d->qpos[i]);

        auto u = ctrl.step(sensors);

        for(int i=0;i<m->nu;i++) d->ctrl[i] = u[i];

        mj_step(m, d);

        if (d->qpos[2] < 0.4) break;  // robot fell
    }

    double dist = d->qpos[0] - x0;

    mj_deleteData(d);

    return dist < 0 ? 0 : dist;
}

int main(int argc, const char** argv){
    if(argc!=2){
        printf("Usage: ./train humanoid.xml\n");
        return 0;
    }

    char err[1000];
    mjModel* m = mj_loadXML(argv[1], 0, err, 1000);
    if(!m){
        printf("Load error: %s\n", err);
        return 0;
    }

    GA ga;
    ga.verbose = true;

    ga.population = 60;
    ga.generation_max = 150;

    ga.mutation_rate = 0.15;
    ga.crossover_fraction = 0.6;

    ga.init_genes = [&](Chromo& c, auto& r){
        for(double& v : c.p) v = -1 + 2*r();
    };

    ga.mutate = [&](const Chromo& b, auto& r, double s){
        Chromo o=b;
        double rad = 0.3*s;
        for(double& v:o.p) v += rad*(r()-r());
        return o;
    };

    ga.crossover = [&](const Chromo&a,const Chromo&b,auto& r){
        Chromo o;
        for(int i=0;i<a.p.size();i++){
            double R=r();
            o.p[i]=R*a.p[i] + (1-R)*b.p[i];
        }
        return o;
    };

    ga.eval_solution = [&](auto& c, auto& res){
        res.fitness = evaluate(c, m);
        return true;
    };

    ga.calculate_SO_total_fitness = [&](auto& X){
        return X.middle_costs.fitness;
    };

    ga.SO_report_generation = [&](int gen, const auto& last, const Chromo& best){
        std::cout << "Gen " << gen
                  << "   Best=" << last.best_total_cost
                  << "   Avg=" << last.average_cost << std::endl;
    };

    ga.solve();

    std::vector<double> best;
    if(ga.last_generation.chromosomes.empty()){
        std::cerr << "Error: last_generation.chromosomes is empty. No solution to save.\n";
        mj_deleteModel(m);
        return 1;
    }
    int best_idx = ga.last_generation.best_chromosome_index;
    if(best_idx < 0 || best_idx >= (int)ga.last_generation.chromosomes.size()){
        std::cerr << "Warning: best_chromosome_index is invalid (" << best_idx << "). Falling back to index 0.\n";
        best = ga.last_generation.chromosomes[0].genes.p;
    } else {
        best = ga.last_generation.chromosomes[best_idx].genes.p;
    }

    std::ofstream fout("best_genome.txt");
    for(double v: best) fout << v << "\n";
    fout.close();

    mj_deleteModel(m);

    return 0;
}