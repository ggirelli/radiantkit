{% if shell_type is not none: %}{{shell_type}}{% endif %}
{% if conda_env is not none:%}conda init
conda activate {{conda_env}}{% endif %}
cd {{root_folder}}
snakemake --configfile {{name}}.config.yaml -s {{name}}.snakefile
