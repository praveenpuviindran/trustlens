"""CLI command for feature extraction."""

import typer
from rich.console import Console
from rich.table import Table
from trustlens.db.session import get_session
from trustlens.services.feature_engineering import FeatureEngineeringService

console = Console()


def extract_features(run_id: str = typer.Option(..., "--run-id", help="Run ID")):
    """Extract features for a run."""
    console.print(f"[bold blue]Extracting features for run_id={run_id}[/bold blue]")
    
    try:
        session = get_session()
        service = FeatureEngineeringService(session)
        feature_count = service.compute_features(run_id)
        console.print(f"[green]✓[/green] Extracted {feature_count} features")
        
        features = service.get_features(run_id)
        grouped = {}
        for f in features:
            if f.feature_group not in grouped:
                grouped[f.feature_group] = []
            grouped[f.feature_group].append(f)
        
        table = Table(title=f"Features for run_id={run_id}")
        table.add_column("Group", style="cyan")
        table.add_column("Feature", style="magenta")
        table.add_column("Value", style="green", justify="right")
        
        for group_name in sorted(grouped.keys()):
            for i, f in enumerate(sorted(grouped[group_name], key=lambda x: x.feature_name)):
                table.add_row(group_name if i == 0 else "", f.feature_name, f"{f.feature_value:.4f}")
        
        console.print(table)
        
    except ValueError as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise


if __name__ == "__main__":
    typer.run(extract_features)