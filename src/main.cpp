#include "abm.h"
#include "argparse.hpp"
#include "config.h"
#include "distributed/cluster_coordinator.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  distributed::ClusterCoordinator coordinator;
  coordinator.Init(argc, argv);

  argparse::ArgumentParser program("abm", "9.0.0");
  program.add_description("CantataS: community-aware agent-based citation simulator");
  program.add_argument("--config").required().help(".ini configuration file (example in docs/)");
  program.add_argument("--resume")
      .default_value(std::string("config"))
      .help("auto: resume if checkpoint_dir has a snapshot; yes: must resume; no: start fresh; "
            "config: follow start_from_checkpoint");
  program.add_argument("--seed").default_value(std::string("")).help("overrides random_seed from the config");
  program.add_argument("--allow-model-change")
      .default_value(false)
      .implicit_value(true)
      .help("resume even if the model parameters differ from the run that wrote the snapshot");
  program.add_argument("--db-path").default_value(std::string("")).help("ignored (kept for old job scripts)");
  program.add_argument("--storage-backend")
      .default_value(std::string(""))
      .help("ignored (kept for old job scripts)");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    if (coordinator.IsMaster()) {
      std::cerr << err.what() << std::endl << program;
    }
    coordinator.Finalize();
    return kExitError;
  }

  try {
    SimulationConfig config = LoadConfig(program.get<std::string>("--config"));
    std::string seed = program.get<std::string>("--seed");
    if (!seed.empty()) {
      config.random_seed = std::stoull(seed);
    }
    config.allow_model_change = program.get<bool>("--allow-model-change");
    std::string resume = program.get<std::string>("--resume");
    ResumeMode mode;
    if (resume == "auto") {
      mode = ResumeMode::Auto;
    } else if (resume == "yes") {
      mode = ResumeMode::Yes;
    } else if (resume == "no") {
      mode = ResumeMode::No;
    } else if (resume == "config") {
      mode = config.start_from_checkpoint ? ResumeMode::Yes : ResumeMode::No;
    } else {
      throw std::runtime_error("--resume must be auto, yes, no or config");
    }
    int code;
    {
      ABM abm(config, coordinator);
      code = abm.Run(mode);
    }
    coordinator.Finalize();
    return code;
  } catch (const std::exception &e) {
    std::cerr << "[rank " << coordinator.Rank() << "] error: " << e.what() << std::endl;
    coordinator.Abort(kExitError);
  }
}
