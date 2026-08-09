"""
Main Experiment Runner
Entry point for running all experiments or specific experiment blocks.
Updated to include Section 5 (universality) and Section 6 (real embeddings).
"""
import argparse
import logging
import sys
import os

from config.global_config import CONFIG, set_seed
from core.reproducibility import setup_logging


def run_all_experiments():
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING ALL EXPERIMENTS")
    logger.info("=" * 80 + "\n")

    experiments = [
        ('norm_concentration',    'Norm Concentration'),
        ('distance_geometry',     'Distance Geometry'),
        ('hubness',               'Hubness Analysis'),
        ('jl_projections',        'Johnson-Lindenstrauss Projections'),
        ('spectral',              'Spectral Analysis'),
        ('scaling_comparison',    'Cross-Phenomenon Scaling'),
        ('universality',          'Section 5: Universality of -0.623 Exponent'),
        ('preprocess_embeddings', 'Section 6 Preprocessing'),
        ('embedding_geometry',    'Section 6: Real Embedding Geometry'),
        ('discrepancy',           'Section 6.4: Discrepancy Analysis'),
        ('prediction_table',      'Section 7: Prediction Table'),
    ]

    for exp_key, exp_name in experiments:
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT: {exp_name}")
        logger.info(f"{'='*80}\n")
        try:
            run_experiment(exp_key)
        except Exception as e:
            logger.error(f"Error in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Continuing with next experiment...")

    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 80)


def run_experiment(experiment_name: str):
    logger = logging.getLogger(__name__)

    if experiment_name == 'norm_concentration':
        from experiments.norm_concentration.run_norm_experiment import run_norm_concentration_experiment
        run_norm_concentration_experiment()
    elif experiment_name == 'distance_geometry':
        from experiments.distance_geometry.run_distance_experiment import run_distance_geometry_experiment
        run_distance_geometry_experiment()
    elif experiment_name == 'hubness':
        from experiments.distance_geometry.hubness_analysis import run_hubness_analysis
        run_hubness_analysis()
    elif experiment_name == 'jl_projections':
        from experiments.jl_projections.run_jl_experiment import run_jl_projection_experiment
        run_jl_projection_experiment()
    elif experiment_name == 'spectral':
        from experiments.spectral_analysis.run_spectral_experiment import run_spectral_analysis_experiment
        run_spectral_analysis_experiment()
    elif experiment_name == 'scaling_comparison':
        from experiments.cross_analysis.run_scaling_comparison import run_scaling_comparison
        run_scaling_comparison()
    elif experiment_name == 'universality':
        from experiments.universality.run_universality_experiment import main as run_universality
        run_universality(fast=False)
    elif experiment_name == 'universality_fast':
        from experiments.universality.run_universality_experiment import main as run_universality
        run_universality(fast=True)
    elif experiment_name == 'preprocess_embeddings':
        import subprocess
        result = subprocess.run(
            [sys.executable, "experiments/real_embeddings/preprocess_embeddings.py", "--skip-cached"],
            check=False
        )
        if result.returncode != 0:
            logger.warning("Preprocessing had errors — some datasets may be missing")
    elif experiment_name == 'embedding_geometry':
        from experiments.real_embeddings.run_embedding_geometry import run_all
        run_all(n_trials=5)
    elif experiment_name == 'embedding_geometry_fast':
        from experiments.real_embeddings.run_embedding_geometry import run_all
        run_all(n_trials=2)
    elif experiment_name == 'discrepancy':
        from experiments.real_embeddings.analyze_discrepancies import run_all_discrepancy_analyses
        run_all_discrepancy_analyses()
    elif experiment_name == 'prediction_table':
        from tables.generate_prediction_table import main as gen_table
        gen_table()
    else:
        logger.error(f"Unknown experiment: {experiment_name}")
        logger.info(
            "Available: norm_concentration, distance_geometry, hubness, "
            "jl_projections, spectral, scaling_comparison, universality, "
            "universality_fast, preprocess_embeddings, embedding_geometry, "
            "embedding_geometry_fast, discrepancy, prediction_table"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='High-Dimensional Geometry Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all
  python main.py --experiment universality
  python main.py --experiment preprocess_embeddings
  python main.py --experiment embedding_geometry
  python main.py --experiment discrepancy
  python main.py --experiment prediction_table
  python main.py --experiment universality_fast
  python main.py --experiment embedding_geometry_fast
  python main.py --list
        """
    )
    parser.add_argument('--experiment', '-e', type=str)
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--list', '-l', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    log_file = os.path.join('results', 'experiment.log')
    setup_logging(log_file=log_file, level=args.log_level)

    if args.seed is not None:
        CONFIG.RANDOM_SEED = args.seed
    set_seed(CONFIG.RANDOM_SEED)

    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to: {CONFIG.RANDOM_SEED}")

    if args.list:
        print("""
Available experiments:

  ORIGINAL (Sections 3-4):
    norm_concentration, distance_geometry, hubness,
    jl_projections, spectral, scaling_comparison

  NEW — Section 5 (Universality):
    universality          Full run (50 trials, ~2hr)
    universality_fast     Quick test (10 trials, ~25min)

  NEW — Section 6 (Real embeddings):
    preprocess_embeddings Run FIRST — one-time setup
    embedding_geometry    Full analysis (5 trials, ~1hr)
    embedding_geometry_fast  Quick (2 trials, ~20min)
    discrepancy           BERT/scRNA/GloVe discrepancy analysis

  NEW — Section 7:
    prediction_table      Unified prediction table + figures
        """)
        return

    if args.all:
        run_all_experiments()
    elif args.experiment:
        run_experiment(args.experiment)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
