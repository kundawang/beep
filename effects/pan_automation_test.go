package effects_test

import (
	"errors"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/effects"
	"github.com/gopxl/beep/v2/internal/testtools"
)

const panTestTol = 1e-12

// constantStreamer emits numSamples samples where every channel equals value. It is a
// beep.StreamSeeker so the pan position is fully deterministic.
func constantStreamer(numSamples int, value float64) beep.StreamSeeker {
	data := make([][2]float64, numSamples)
	for i := range data {
		data[i] = [2]float64{value, value}
	}
	return testtools.NewDataStreamer(data)
}

// constantPowerGains returns the expected constant-power left/right gains for a pan value.
func constantPowerGains(pan float64) (gainL, gainR float64) {
	theta := (pan + 1) * math.Pi / 4
	return math.Cos(theta), math.Sin(theta)
}

// collectInChunks drains s using buffers of the given sizes and returns the concatenated output.
// The chunk sizes must cover at least the full length of the test source.
func collectInChunks(s beep.Streamer, chunkSizes ...int) [][2]float64 {
	var out [][2]float64
	for _, size := range chunkSizes {
		buf := make([][2]float64, size)
		n, ok := s.Stream(buf)
		out = append(out, buf[:n]...)
		if !ok {
			break
		}
	}
	return out
}

// A car crossing from left to right: pan sweeps linearly from -1 to +1 over 100 samples.
func carCrossingPoints() []effects.PanPoint {
	return []effects.PanPoint{
		{Position: 0, Pan: -1},
		{Position: 100, Pan: 1},
	}
}

func TestPanAutomation_EndpointsAndMidpoint(t *testing.T) {
	s := effects.NewPanAutomation(constantStreamer(101, 1), carCrossingPoints())

	got := testtools.Collect(s)
	require.Len(t, got, 101)

	// Hard left at the start: the right channel is silent.
	gainL, gainR := constantPowerGains(-1)
	assert.InDelta(t, gainL, got[0][0], panTestTol)
	assert.InDelta(t, gainR, got[0][1], panTestTol)
	assert.InDelta(t, 1.0, got[0][0], panTestTol)
	assert.InDelta(t, 0.0, got[0][1], panTestTol)

	// Center: both channels at sqrt(2)/2 (-3 dB).
	gainL, gainR = constantPowerGains(0)
	assert.InDelta(t, gainL, got[50][0], panTestTol)
	assert.InDelta(t, gainR, got[50][1], panTestTol)
	assert.InDelta(t, math.Sqrt2/2, got[50][0], panTestTol)
	assert.InDelta(t, math.Sqrt2/2, got[50][1], panTestTol)

	// Hard right at the end: the left channel is silent.
	gainL, gainR = constantPowerGains(1)
	assert.InDelta(t, gainL, got[100][0], panTestTol)
	assert.InDelta(t, gainR, got[100][1], panTestTol)
	assert.InDelta(t, 0.0, got[100][0], panTestTol)
	assert.InDelta(t, 1.0, got[100][1], panTestTol)
}

func TestPanAutomation_ConstantEnergyAtMidpoint(t *testing.T) {
	// The pan sweep must never make the sound noticeably quieter: for a unit mono input the
	// summed energy gainL^2 + gainR^2 has to equal 1 at every sample position.
	s := effects.NewPanAutomation(constantStreamer(101, 1), carCrossingPoints())

	got := testtools.Collect(s)
	require.Len(t, got, 101)
	for i, sample := range got {
		energy := sample[0]*sample[0] + sample[1]*sample[1]
		assert.InDelta(t, 1.0, energy, panTestTol, "energy not constant at sample %d", i)
	}
}

func TestPanAutomation_InterpolatedPanValues(t *testing.T) {
	// With a unit input, the output channels equal the gains, so the whole curve can be checked
	// sample by sample against linear pan interpolation.
	s := effects.NewPanAutomation(constantStreamer(101, 1), carCrossingPoints())

	got := testtools.Collect(s)
	require.Len(t, got, 101)
	for i := range got {
		pan := -1 + 2*float64(i)/100
		gainL, gainR := constantPowerGains(pan)
		assert.InDelta(t, gainL, got[i][0], panTestTol, "left gain mismatch at sample %d", i)
		assert.InDelta(t, gainR, got[i][1], panTestTol, "right gain mismatch at sample %d", i)
	}
}

func TestPanAutomation_BlockSizeIndependent(t *testing.T) {
	// The same audio streamed with different chunk sizes must produce identical samples,
	// including across block boundaries in the middle of the pan sweep.
	points := carCrossingPoints()

	reference := testtools.Collect(effects.NewPanAutomation(constantStreamer(101, 1), points))

	for _, chunks := range [][]int{
		{1, 1, 1, 1, 1, 96},
		{7, 13, 1, 29, 51},
		{33, 34, 34},
		{101},
		{64, 64},
		{50, 1, 50},
	} {
		got := collectInChunks(effects.NewPanAutomation(constantStreamer(101, 1), points), chunks...)
		testtools.AssertSamplesEqual(t, reference, got)
	}
}

func TestPanAutomation_SeekRestoresCurvePosition(t *testing.T) {
	reference := testtools.Collect(effects.NewPanAutomation(constantStreamer(101, 1), carCrossingPoints()))

	// Seek straight into the middle of the sweep, then stream the rest.
	s := effects.NewPanAutomation(constantStreamer(101, 1), carCrossingPoints()).(beep.StreamSeeker)
	require.NoError(t, s.Seek(50))
	assert.Equal(t, 50, s.Position())
	got := testtools.Collect(s)
	testtools.AssertSamplesEqual(t, reference[50:], got)

	// Seek backwards: the curve state must be rebuilt from the target position.
	require.NoError(t, s.Seek(10))
	got = testtools.Collect(s)
	testtools.AssertSamplesEqual(t, reference[10:], got)

	// Seek to a keyframe position exactly.
	require.NoError(t, s.Seek(0))
	got = testtools.CollectNum(3, s)
	testtools.AssertSamplesEqual(t, reference[:3], got)

	// Seek past the last keyframe: the final pan value is held.
	require.NoError(t, s.Seek(100))
	got = testtools.Collect(s)
	testtools.AssertSamplesEqual(t, reference[100:], got)
}

func TestPanAutomation_HoldsPanOutsideKeyframes(t *testing.T) {
	points := []effects.PanPoint{
		{Position: 10, Pan: -0.5},
		{Position: 20, Pan: 0.5},
	}
	reference := testtools.Collect(effects.NewPanAutomation(constantStreamer(31, 1), points))

	// Before the first keyframe, the first pan value is held.
	gainL, gainR := constantPowerGains(-0.5)
	for i := 0; i < 10; i++ {
		assert.InDelta(t, gainL, reference[i][0], panTestTol)
		assert.InDelta(t, gainR, reference[i][1], panTestTol)
	}
	// After the last keyframe, the last pan value is held.
	gainL, gainR = constantPowerGains(0.5)
	for i := 20; i <= 30; i++ {
		assert.InDelta(t, gainL, reference[i][0], panTestTol)
		assert.InDelta(t, gainR, reference[i][1], panTestTol)
	}
}

func TestPanAutomation_NormalizationRules(t *testing.T) {
	// Unordered, duplicate, out-of-range and NaN inputs.
	points := []effects.PanPoint{
		{Position: 20, Pan: 5},         // clamp to +1
		{Position: -5, Pan: -5},        // position clamp to 0, pan clamp to -1
		{Position: 10, Pan: 0},         // out of order
		{Position: 0, Pan: math.NaN()}, // NaN -> 0, duplicate position 0 keeps this value
	}
	s := effects.NewPanAutomation(constantStreamer(21, 1), points)
	got := testtools.Collect(s)

	gainL, gainR := constantPowerGains(0) // NaN entry at position 0 wins
	assert.InDelta(t, gainL, got[0][0], panTestTol)
	assert.InDelta(t, gainR, got[0][1], panTestTol)

	gainL, gainR = constantPowerGains(0) // explicit center keyframe
	assert.InDelta(t, gainL, got[10][0], panTestTol)
	assert.InDelta(t, gainR, got[10][1], panTestTol)

	gainL, gainR = constantPowerGains(1) // clamped hard right
	assert.InDelta(t, gainL, got[20][0], panTestTol)
	assert.InDelta(t, gainR, got[20][1], panTestTol)
	assert.InDelta(t, 0.0, got[20][0], panTestTol)
}

func TestPanAutomation_EmptyCurveIsPassthrough(t *testing.T) {
	src, data := testtools.RandomDataStreamer(1000)
	got := testtools.Collect(effects.NewPanAutomation(src, nil))
	testtools.AssertSamplesEqual(t, data, got)

	src2, data2 := testtools.RandomDataStreamer(1000)
	got = testtools.Collect(effects.NewPanAutomation(src2, []effects.PanPoint{}))
	testtools.AssertSamplesEqual(t, data2, got)
}

func TestPanAutomation_SingleKeyframe(t *testing.T) {
	// One keyframe applies a constant pan to the whole stream.
	s := effects.NewPanAutomation(constantStreamer(5, 1), []effects.PanPoint{{Position: 2, Pan: -1}})
	got := testtools.Collect(s)
	for i := range got {
		assert.InDelta(t, 1.0, got[i][0], panTestTol, "sample %d", i)
		assert.InDelta(t, 0.0, got[i][1], panTestTol, "sample %d", i)
	}
}

func TestPanAutomation_MonoInput(t *testing.T) {
	// Mono material (identical channels) follows the same law as stereo input.
	s := effects.NewPanAutomation(constantStreamer(101, 0.75), carCrossingPoints())
	got := testtools.Collect(s)
	for i := range got {
		pan := -1 + 2*float64(i)/100
		gainL, gainR := constantPowerGains(pan)
		assert.InDelta(t, 0.75*gainL, got[i][0], panTestTol)
		assert.InDelta(t, 0.75*gainR, got[i][1], panTestTol)
	}
}

func TestPanAutomation_DoesNotMutateInputPoints(t *testing.T) {
	points := []effects.PanPoint{
		{Position: 20, Pan: 5},
		{Position: -5, Pan: -5},
	}
	orig := append([]effects.PanPoint(nil), points...)
	_ = effects.NewPanAutomation(constantStreamer(1, 1), points)
	assert.Equal(t, orig, points)
}

func TestPanAutomation_PropagatesStreamerError(t *testing.T) {
	expectedErr := errors.New("boom")
	s := effects.NewPanAutomation(testtools.NewErrorStreamer(expectedErr), carCrossingPoints())

	buf := make([][2]float64, 8)
	n, ok := s.Stream(buf)
	assert.Equal(t, 0, n)
	assert.False(t, ok)
	assert.Same(t, expectedErr, s.Err())
}

func TestPanAutomation_PropagatesDelayedError(t *testing.T) {
	expectedErr := errors.New("late boom")
	src := testtools.NewDelayedErrorStreamer(constantStreamer(10, 1), 4, expectedErr)
	s := effects.NewPanAutomation(src, carCrossingPoints())

	got := testtools.Collect(s)
	assert.Len(t, got, 4)
	assert.Same(t, expectedErr, s.Err())
}

func TestPanAutomation_SeekErrorIsTransparent(t *testing.T) {
	expectedErr := errors.New("seek refused")
	src := testtools.NewSeekErrorStreamer(constantStreamer(10, 1), expectedErr)
	s := effects.NewPanAutomation(src, carCrossingPoints()).(beep.StreamSeeker)

	assert.Same(t, expectedErr, s.Seek(5))
	assert.Equal(t, 0, s.Position())
}

func TestPanAutomation_NonSeekableSource(t *testing.T) {
	src := beep.StreamerFunc(func(samples [][2]float64) (n int, ok bool) {
		for i := range samples {
			samples[i] = [2]float64{1, 1}
		}
		return len(samples), true
	})
	s := effects.NewPanAutomation(src, carCrossingPoints())

	_, ok := s.(beep.StreamSeeker)
	assert.False(t, ok, "a non-seekable source must not expose StreamSeeker")

	got := testtools.CollectNum(101, s)
	require.Len(t, got, 101)
	gainL, gainR := constantPowerGains(0)
	assert.InDelta(t, gainL, got[50][0], panTestTol)
	assert.InDelta(t, gainR, got[50][1], panTestTol)
}

// TestPan covers the pre-existing static Pan effect to guard its documented behavior.
func TestPan(t *testing.T) {
	src, data := testtools.RandomDataStreamer(500)
	pan := &effects.Pan{Streamer: src, Pan: 0}
	testtools.AssertSamplesEqual(t, data, testtools.Collect(pan))

	for _, v := range []float64{-1, -0.25, 0.25, 1} {
		src, data = testtools.RandomDataStreamer(500)
		pan = &effects.Pan{Streamer: src, Pan: v}
		got := testtools.Collect(pan)
		for i := range data {
			var wantL, wantR float64
			switch {
			case v < 0:
				wantL = data[i][0] + -v*data[i][1]
				wantR = data[i][1] - -v*data[i][1]
			case v > 0:
				wantL = data[i][0] - v*data[i][0]
				wantR = data[i][1] + v*data[i][0]
			default:
				wantL, wantR = data[i][0], data[i][1]
			}
			assert.InDelta(t, wantL, got[i][0], 1e-12, "pan %v left at %d", v, i)
			assert.InDelta(t, wantR, got[i][1], 1e-12, "pan %v right at %d", v, i)
		}
		assert.NoError(t, pan.Err())
	}

	// Errors propagate.
	expectedErr := errors.New("pan boom")
	pan = &effects.Pan{Streamer: testtools.NewErrorStreamer(expectedErr)}
	assert.Same(t, expectedErr, pan.Err())
}
