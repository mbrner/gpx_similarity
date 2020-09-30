import click
import toml


@click.group()
def cli():
    pass


@cli.command()  # @cli, not @click!
@click.argument('config', type=click.Path(exists=True))
@click.argument('in_files', nargs=-1)
@click.option('--type-extraction/--no-type-extraction', 'type_extraction', default=False)
@click.option('--absolute-paths/--relativ-paths', 'expand_paths', default=False)
@click.option('-d', '--dataset-name', 'dataset_name', default='unknown')
@click.option('-r', '--route-type', 'route_type', default='unknown')
def add_train_files(type_extraction, config, dataset_name, route_type, in_files, expand_paths):
    from src.create_figs import add_train_files
    if len(in_files) > 3:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}',... '{in_files[-1]}') [{len(in_files)} files]"
    else:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}', '{in_files[2]}') [{len(in_files)} files]"
    msg = [f'Adding files: {in_files_str} to train dataset: "{dataset_name}"']
    msg.append(f'\tconfig: {config}')
    msg.append(f'\textract route_type: {type_extraction} [default="{route_type}"]')
    click.echo('\n'.join(msg))
    config = toml.load(config)
    add_train_files(config=config,
                    in_files=in_files,
                    dataset_name=dataset_name,
                    default_route_type=route_type,
                    extract_route_type=type_extraction,
                    expand_paths=expand_paths)


@cli.command()  # @cli, not @click!
@click.argument('config', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('-w', '--weights', 'weights', default=None)
def train_model(config, output_dir, weights):
    from src.nn import train
    config = toml.load(config)
    train(config, output_dir, weights)


@cli.command()
@click.argument('config', type=click.Path(exists=True))
@click.argument('reference_database', type=click.Path())
@click.argument('in_files', nargs=-1)
@click.option('--type-extraction/--no-type-extraction', 'type_extraction', default=False)
@click.option('--absolute-paths/--relativ-paths', 'expand_paths', default=False)
@click.option('-d', '--dataset-name', 'dataset_name', default='unknown')
@click.option('-r', '--route-type', 'route_type', default='unknown')
def add_reference_files(type_extraction, config, dataset_name, route_type, in_files, reference_database, expand_paths):
    from src.create_figs import add_reference_files
    if len(in_files) > 3:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}',... '{in_files[-1]}') [{len(in_files)} files]"
    else:
        in_files_str = f"('{in_files[0]}', '{in_files[1]}', '{in_files[2]}') [{len(in_files)} files]"
    msg = [f'Adding reference files: {in_files_str} to dataset "{dataset_name}" in database {reference_database}']
    msg.append(f'\tconfig: {config}')
    msg.append(f'\textract route_type: {type_extraction} [default="{route_type}"]')
    click.echo('\n'.join(msg))
    config = toml.load(config)
    add_reference_files(config=config,
                        in_files=in_files,
                        reference_database=reference_database,
                        dataset_name=dataset_name,
                        default_route_type=route_type,
                        extract_route_type=type_extraction,
                        expand_paths=expand_paths)



@cli.command()  # @cli, not @click!
@click.argument('config', type=click.Path(exists=True))
@click.argument('reference_database', type=click.Path())
@click.argument('weights', type=click.Path())
@click.argument('output_dir', type=click.Path())
def apply_model_to_reference_files(config, output_dir, reference_database, weights):
    from src.nn import apply_model_ref_files
    config = toml.load(config)
    apply_model_ref_files(config, output_dir, reference_database, weights)

if __name__ == '__main__':
    cli()



