import click


@click.group()
def cli():
    pass


@cli.command()  # @cli, not @click!
@click.argument('config', type=click.Path(exists=True))
@click.argument('in_files', nargs=-1)
@click.option('--type-extraction/--no-type-extraction', 'type_extraction', default=False)
@click.option('-d', '--dataset-name', 'dataset_name', default='unknown')
@click.option('-r', '--route-type', 'route_type', default='unknown')
def add_train_dataset(type_extraction, config, dataset_name, route_type, in_files):
    from src.create_figs import add_train_files
    import toml
    import redis
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
                    extract_route_type=type_extraction)



if __name__ == '__main__':
    cli()



