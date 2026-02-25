
n=8

python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="FirstHalf" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="NonReziprok" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="AortaIndex" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="AortaIndexGuard" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="CrossIndex" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="CrossIndexGuard" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="FreqIndex" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="AutocorrIndex" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="AortaIndexGuard" --modelselection="chasel"

python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="NonInjection" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="NonInjectionGuard" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="NonInjectionMid" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="VisualIndex" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="VisualIndexGuard" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="BolusIndex" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="SingleBolusIndex" --modelselection="chasel"
python train_model1.py --number $n --resampleparas=True --loss="mae"  --epochs=150 --batchsize=32 --normaorta="fixed" --config="/config_model10_Lin.json" --sUseIndex="none" --modelselection="chasel"

