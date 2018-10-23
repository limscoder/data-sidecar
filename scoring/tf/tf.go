package tf

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

// example tf arg
// $HOME/projects/predictatron/models/lstm/frozen-tf/model-btc_usd-5m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-btc_usd-15m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-btc_usd-60m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-eth_usd-5m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-eth_usd-15m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-eth_usd-60m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-bch_usd-5m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-bch_usd-15m:$HOME/projects/predictatron/models/lstm/frozen-tf/model-bch_usd-60m

// ModelMetadata is paramaters for a Model
type ModelMetadata struct {
	ModelKey              string   `json:"model_key"`
	InputSteps            int      `json:"input_steps"`
	InputOperation        string   `json:"input_operation"`
	OutputSteps           int      `json:"output_steps"`
	OutputOperation       string   `json:"output_operation"`
	PredictFutureDuration int      `json:"predict_future_duration"`
	Target                string   `json:"target"`
	Columns               []string `json:"columns"`
}

// Model is a TensorFlow model and related metadata
type Model struct {
	Model    *tensorflow.SavedModel
	Metadata *ModelMetadata
}

// LoadModels loads Models from disk
func LoadModels(tfpath string) ([]Model, error) {
	models := make([]Model, 0, 5)
	if tfpath != "" {
		paths := strings.Split(tfpath, ":")
		for _, path := range paths {
			info, err := os.Stat(path)
			if err != nil || !info.IsDir() {
				return nil, fmt.Errorf("invalid model path: %s", path)
			}

			parts := strings.Split(path, "/")
			modelKey := parts[len(parts)-1]
			model, err := tensorflow.LoadSavedModel(path, []string{modelKey}, nil)
			if err != nil {
				return nil, fmt.Errorf("error loading tensorflow model: %v", err)
			}

			metaFile := fmt.Sprintf("%s/params.json", path)
			content, err := ioutil.ReadFile(metaFile)
			if err != nil {
				return nil, fmt.Errorf("error loading tensorflow metadata: %v", err)
			}
			meta := &ModelMetadata{}
			err = json.Unmarshal(content, meta)
			if err != nil {
				return nil, fmt.Errorf("error deserializating tensorflow metadata: %v", err)
			}

			models = append(models, Model{Model: model, Metadata: meta})
		}
	}

	return models, nil
}
