package tf

import (
	"fmt"
	"log"
	"time"

	"github.com/open-fresh/data-sidecar/util"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type scaler struct {
	min      float32
	max      float32
	diff     float32
	scale    float32
	scaleMin float32
	data     []util.DataPoint
}

func newScaler(data []util.DataPoint) *scaler {
	min := data[0].Val
	max := data[0].Val
	for _, point := range data {
		if point.Val < min {
			min = point.Val
		}
		if point.Val > max {
			max = point.Val
		}
	}
	diff := max - min
	scale := 1. / diff
	scaleMin := 0 - min*scale
	return &scaler{
		min:      float32(min),
		max:      float32(max),
		diff:     float32(diff),
		scale:    float32(scale),
		scaleMin: float32(scaleMin),
		data:     data,
	}
}

func (s *scaler) normalize() []float32 {
	normalized := make([]float32, len(s.data), len(s.data))
	for idx, point := range s.data {
		normalized[idx] = float32(point.Val)*s.scale + s.scaleMin
	}
	return normalized
}

func (s *scaler) denormalize(val float32) float32 {
	return (val - s.scaleMin) / s.scale
}

// Scorer scores series with multi-variate models
type Scorer struct {
	maxPoints  int
	datapoints map[string][]util.DataPoint
	models     []Model
	recorder   util.Recorder
}

// NewScorer instantiates a Scorer
func NewScorer(maxPoints int, models []Model, recorder util.Recorder) *Scorer {
	return &Scorer{
		maxPoints:  maxPoints,
		datapoints: make(map[string][]util.DataPoint),
		models:     models,
		recorder:   recorder,
	}
}

func (s *Scorer) inputTensor(model Model) (*tensorflow.Tensor, error) {
	input := []float32{}
	for _, col := range model.Metadata.Columns {
		data, exists := s.datapoints[col]
		if !exists {
			return nil, fmt.Errorf("missing datapoints for: %s", col)
		}
		startIdx := len(data) - model.Metadata.InputSteps
		steps := data[startIdx:]
		scaler := newScaler(steps)
		input = append(input, scaler.normalize()...)
	}

	tensor2d := [][]float32{input}
	tensor3d := [][][]float32{tensor2d}
	return tensorflow.NewTensor(tensor3d)
}

// Append adds new datapoints to a series
func (s *Scorer) Append(labels map[string]string, data []util.DataPoint) {
	// TODO: handle targets with multiple series
	// seriesKey := util.MapSSToS(labels)
	seriesKey := labels["__name__"]
	if _, exists := s.datapoints[seriesKey]; !exists {
		s.datapoints[seriesKey] = []util.DataPoint{}
	}

	newPointCount := len(data)
	if newPointCount >= s.maxPoints {
		startIdx := newPointCount - s.maxPoints
		s.datapoints[seriesKey] = data[startIdx:]
	} else {
		datapoints := s.datapoints[seriesKey]
		startIdx := len(datapoints) + newPointCount - s.maxPoints
		if startIdx < 0 {
			startIdx = 0
		}
		s.datapoints[seriesKey] = append(datapoints[startIdx:], data...)
	}
}

// ScoreModels runs all models and scores the result
func (s *Scorer) ScoreModels() {
	for _, model := range s.models {
		if _, exists := s.datapoints[model.Metadata.Target]; exists && len(s.datapoints[model.Metadata.Target]) > model.Metadata.InputSteps {
			// format input
			input, err := s.inputTensor(model)
			if err != nil {
				log.Println("failed to generate input tensor", err)
				break
			}

			// run model prediction
			inputOp := model.Model.Graph.Operation(model.Metadata.InputOperation)
			outputOp := model.Model.Graph.Operation(model.Metadata.OutputOperation)
			result, err := model.Model.Session.Run(
				map[tensorflow.Output]*tensorflow.Tensor{
					inputOp.Output(0): input,
				},
				[]tensorflow.Output{
					outputOp.Output(0),
				},
				nil,
			)
			if err != nil {
				log.Println("failed to run model", err)
			}

			// format output
			targetData := s.datapoints[model.Metadata.Target]
			targetStart := len(targetData) - model.Metadata.InputSteps
			targetSteps := targetData[targetStart:]
			targetScaler := newScaler(targetSteps)
			targetResult := result[0].Value().([][]float32)
			targetPrediction := targetScaler.denormalize(targetResult[0][0])

			// expose value to endpoint
			dataPoint := util.DataPoint{Val: float64(targetPrediction), Time: time.Now().Unix()}
			labels := map[string]string{
				"__name__":         fmt.Sprintf("predict_sidecar:%s", model.Metadata.Target),
				"predict_duration": fmt.Sprintf("%v", model.Metadata.PredictFutureDuration),
			}
			s.recorder.Record(util.Metric{Desc: labels, Data: dataPoint})

		}
	}
}
