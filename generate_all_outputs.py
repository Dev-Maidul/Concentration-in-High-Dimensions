"""
Generate All Outputs
Orchestrates running all experiments, generating all figures and tables.
"""
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.global_config import CONFIG, set_seed
from core.reproducibility import setup_logging


def generate_all_outputs(skip_experiments: bool = False):
    """
    Generate all outputs for the research paper.
    
    Parameters
    ----------
    skip_experiments : bool
        If True, skip running experiments and only generate figures/tables
        from existing results
    """
    setup_logging(log_file='results/generate_all_outputs.log')
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("HIGH-DIMENSIONAL GEOMETRY PROJECT")
    logger.info("COMPLETE OUTPUT GENERATION PIPELINE")
    logger.info("=" * 80 + "\n")
    
    # Set random seed for reproducibility
    set_seed(CONFIG.RANDOM_SEED)
    logger.info(f"Random seed: {CONFIG.RANDOM_SEED}\n")
    
    # Step 1: Run all experiments (if not skipping)
    if not skip_experiments:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: RUNNING ALL EXPERIMENTS")
        logger.info("=" * 80 + "\n")
        
        from main import run_all_experiments
        run_all_experiments()
    else:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: SKIPPING EXPERIMENTS (using existing results)")
        logger.info("=" * 80 + "\n")
    
    # Step 2: Generate all figures
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: GENERATING ALL FIGURES")
    logger.info("=" * 80 + "\n")
    
    from visualization.generate_all_figures import generate_all_figures
    generate_all_figures()
    
    # Step 3: Generate all tables
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: GENERATING ALL TABLES")
    logger.info("=" * 80 + "\n")
    
    from tables.generate_all_tables import generate_all_tables
    generate_all_tables()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE OUTPUT GENERATION FINISHED")
    logger.info("=" * 80 + "\n")
    
    logger.info("Output locations:")
    logger.info(f"  Raw results:  {CONFIG.RESULTS_RAW}")
    logger.info(f"  Figures:      {CONFIG.RESULTS_FIGURES}")
    logger.info(f"  Tables:       {CONFIG.RESULTS_TABLES}")
    
    logger.info("\nAll outputs have been generated successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all experimental outputs')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip running experiments, only generate figures/tables')
    parser.add_argument('--figures-only', action='store_true',
                       help='Only generate figures from existing results')
    parser.add_argument('--tables-only', action='store_true',
                       help='Only generate tables from existing results')
    
    args = parser.parse_args()
    
    if args.figures_only:
        setup_logging()
        from visualization.generate_all_figures import generate_all_figures
        generate_all_figures()
    elif args.tables_only:
        setup_logging()
        from tables.generate_all_tables import generate_all_tables
        generate_all_tables()
    else:
        generate_all_outputs(skip_experiments=args.skip_experiments)
