#!/bin/bash
# 🧬 Example Surf2Spot Commands (As Originally Intended)

# Set up working directory
cd /home/sferrier/epitope_mapping/Surf2Spot
export SURF2SPOT_DIR=$(pwd)

# Environment paths
export SURF2SPOT_ENV="/home/sferrier/miniforge3/envs/surf2spot"
export SURF2SPOT_TOOLS_ENV="/home/sferrier/miniforge3/envs/surf2spot_tools"

# Data paths
export INPUT_DIR="${SURF2SPOT_DIR}/test_sabdab_example/input"
export OUTPUT_DIR="${SURF2SPOT_DIR}/test_sabdab_example"
export PREDICT_DIR="${SURF2SPOT_DIR}/test_sabdab_example/predict"

# Create directories
mkdir -p $INPUT_DIR $PREDICT_DIR

echo "🚀 Running Surf2Spot Pipeline (Original Commands)"
echo "=================================================="

# Step 1: NB-preprocess
echo "📊 Step 1: NB-preprocess"
/home/sferrier/miniforge3/micromamba run -p $SURF2SPOT_ENV \
    python Surf2Spot/main.py NB-preprocess \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ NB-preprocess completed"
else
    echo "❌ NB-preprocess failed"
    exit 1
fi

# Step 2: NB-craft
echo "📊 Step 2: NB-craft"
/home/sferrier/miniforge3/micromamba run -p $SURF2SPOT_ENV \
    python Surf2Spot/main.py NB-craft \
    -i $OUTPUT_DIR \
    -s ${OUTPUT_DIR}/seq.fasta \
    -emb ${OUTPUT_DIR}/seq_prottrans.h5 \
    -ds ${OUTPUT_DIR}/chainsaw.tsv

if [ $? -eq 0 ]; then
    echo "✅ NB-craft completed"
else
    echo "❌ NB-craft failed"
    exit 1
fi

# Step 3: NB-predict
echo "📊 Step 3: NB-predict"
/home/sferrier/miniforge3/micromamba run -p $SURF2SPOT_ENV \
    python Surf2Spot/main.py NB-predict \
    -i $OUTPUT_DIR \
    -o $PREDICT_DIR \
    -emb ${OUTPUT_DIR}/seq_prottrans.h5 \
    --model model/NB/model.pt \
    --threshold 0.4

if [ $? -eq 0 ]; then
    echo "✅ NB-predict completed"
else
    echo "❌ NB-predict failed"
    exit 1
fi

# Step 4: NB-draw
echo "📊 Step 4: NB-draw"
/home/sferrier/miniforge3/micromamba run -p $SURF2SPOT_ENV \
    python Surf2Spot/main.py NB-draw \
    -i $OUTPUT_DIR \
    -o $PREDICT_DIR

if [ $? -eq 0 ]; then
    echo "✅ NB-draw completed"
    echo "🎉 Surf2Spot pipeline completed successfully!"
else
    echo "❌ NB-draw failed"
    exit 1
fi

# Show results
echo ""
echo "📋 Results Summary:"
echo "==================="
echo "Input files: $(ls $INPUT_DIR/*.pdb 2>/dev/null | wc -l) PDB files"
echo "Chainsaw domains: $(ls ${OUTPUT_DIR}/chainsaw.tsv 2>/dev/null | wc -l) TSV file"
echo "Surface files: $(ls ${OUTPUT_DIR}/*_all_5.0.ply 2>/dev/null | wc -l) PLY files"
echo "Prediction files: $(ls ${PREDICT_DIR}/*.csv 2>/dev/null | wc -l) CSV files"
echo ""
echo "🔗 View results:"
echo "   CSV files: $PREDICT_DIR/*.csv"
echo "   PyMOL files: $PREDICT_DIR/*.pse"
echo ""