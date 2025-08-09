from data.dataset_preparer import prepare_dataset
from train.train_sft import train_sft
from eval.eval_suite import run_eval_suite

if __name__ == "__main__":
    R_D = prepare_dataset()

    R_T = train_sft(train_path=R_D["train_path"], eval_path=R_D["eval_path"])

    print(str(R_T))

    run_eval_suite(adapters=str(R_T))
