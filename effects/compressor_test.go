package effects_test

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/effects"
	"github.com/gopxl/beep/v2/internal/testtools"
)

const (
	testSR            = beep.SampleRate(44100)
	testThresholdDb   = -20.0
	testRatio         = 4.0
	testMakeupDb      = 0.0
	instant           = time.Duration(0)
	expectedReduction = -15.0 // (threshold - level) * (1 - 1/ratio) for level 0 dB, T=-20 dB, R=4
	compressedUnity   = 0.1778279410038923
	epsilon           = 1e-12
)

func newTestCompressor(s beep.Streamer) *effects.Compressor {
	c, err := effects.NewCompressor(s, testSR, testThresholdDb, testRatio, instant, instant, testMakeupDb)
	if err != nil {
		panic(err)
	}
	return c
}

func constantStreamer(n int, v float64) beep.StreamSeeker {
	data := make([][2]float64, n)
	for i := range data {
		data[i] = [2]float64{v, v}
	}
	return testtools.NewDataStreamer(data)
}

// collectChunks drains total samples from s in chunks of the given size.
func collectChunks(s beep.Streamer, total, chunk int) [][2]float64 {
	out := make([][2]float64, 0, total)
	buf := make([][2]float64, chunk)
	for len(out) < total {
		n, ok := s.Stream(buf)
		out = append(out, buf[:n]...)
		if !ok {
			break
		}
	}
	return out
}

func assertSamplesClose(t *testing.T, expected, actual [][2]float64) {
	t.Helper()
	if assert.Len(t, actual, len(expected)) {
		for i := range expected {
			assert.InDelta(t, expected[i][0], actual[i][0], epsilon, "left sample %d", i)
			assert.InDelta(t, expected[i][1], actual[i][1], epsilon, "right sample %d", i)
		}
	}
}

// The zero value is a valid streamer and passes samples through unchanged.
func TestCompressor_ZeroValueIsPassThrough(t *testing.T) {
	s, data := testtools.RandomDataStreamer(64)
	c := &effects.Compressor{Streamer: s}

	got := testtools.Collect(c)
	assertSamplesClose(t, data, got)
	assert.NoError(t, c.Err())
	assert.Zero(t, c.GainReduction())
}

// Ratio 1 and zero makeup never alter the signal, even above the threshold.
func TestCompressor_UnitySettingsAreTransparent(t *testing.T) {
	s := constantStreamer(16, 1.0)
	c, err := effects.NewCompressor(s, testSR, testThresholdDb, 1, instant, instant, 0)
	assert.NoError(t, err)

	got := testtools.Collect(c)
	for i, g := range got {
		assert.InDelta(t, 1.0, g[0], epsilon, "sample %d", i)
		assert.InDelta(t, 1.0, g[1], epsilon, "sample %d", i)
	}
	assert.Zero(t, c.GainReduction())
}

// With no makeup gain the compressor only attenuates, so normal [-1, 1] samples are never
// pushed out of range or clipped.
func TestCompressor_DoesNotClipNormalSamples(t *testing.T) {
	s, data := testtools.RandomDataStreamer(512)
	c := newTestCompressor(s)

	got := testtools.Collect(c)
	if assert.Len(t, got, len(data)) {
		for i := range data {
			assert.LessOrEqual(t, math.Abs(got[i][0]), math.Abs(data[i][0])+epsilon)
			assert.LessOrEqual(t, math.Abs(got[i][1]), math.Abs(data[i][1])+epsilon)
			assert.LessOrEqual(t, math.Abs(got[i][0]), 1.0)
			assert.LessOrEqual(t, math.Abs(got[i][1]), 1.0)
		}
	}
}

// Makeup gain with unity ratio is a pure static gain.
func TestCompressor_MakeupGainOnly(t *testing.T) {
	s := constantStreamer(8, 0.5)
	c, err := effects.NewCompressor(s, testSR, testThresholdDb, 1, instant, instant, 6)
	assert.NoError(t, err)

	gain := math.Pow(10, 6.0/20)
	got := testtools.Collect(c)
	for i, g := range got {
		assert.InDelta(t, 0.5*gain, g[0], epsilon, "sample %d", i)
		assert.InDelta(t, 0.5*gain, g[1], epsilon, "sample %d", i)
	}
}

// A constant signal above the threshold reaches the textbook steady state:
// output dB = threshold + (level dB - threshold) / ratio, reached on the first sample with an
// instant attack, and held there.
func TestCompressor_ConstantAmplitudeGainCurve(t *testing.T) {
	s := constantStreamer(32, 1.0)
	c := newTestCompressor(s)

	got := testtools.Collect(c)
	for i, g := range got {
		assert.InDelta(t, compressedUnity, g[0], epsilon, "sample %d", i)
		assert.InDelta(t, compressedUnity, g[1], epsilon, "sample %d", i)
	}
	assert.InDelta(t, expectedReduction, c.GainReduction(), epsilon)
}

// Signals at or below the threshold are not compressed at all.
func TestCompressor_BelowThresholdUntouched(t *testing.T) {
	thresholdLin := math.Pow(10, testThresholdDb/20)
	s := constantStreamer(8, thresholdLin)
	c := newTestCompressor(s)

	got := testtools.Collect(c)
	for i, g := range got {
		assert.InDelta(t, thresholdLin, g[0], epsilon, "sample %d", i)
		assert.InDelta(t, thresholdLin, g[1], epsilon, "sample %d", i)
	}
	assert.Zero(t, c.GainReduction())
}

// A step into a loud signal follows the attack coefficient; the recovery follows the release
// coefficient. Both are the exact one-pole recurrence envelopeCoefficient implements.
func TestCompressor_StepResponseAttackAndRelease(t *testing.T) {
	sr := beep.SampleRate(100)
	release := time.Second
	data := make([][2]float64, 8)
	data[0] = [2]float64{1, 1} // single loud step, rest is silence
	s := testtools.NewDataStreamer(data)

	c, err := effects.NewCompressor(s, sr, testThresholdDb, testRatio, instant, release, 0)
	assert.NoError(t, err)

	releaseCoef := math.Exp(-1 / (float64(sr) * release.Seconds()))
	wantEnv := expectedReduction
	buf := make([][2]float64, 1)
	for i := range data {
		n, ok := c.Stream(buf)
		assert.Equal(t, 1, n)
		assert.True(t, ok)
		assert.InDelta(t, wantEnv, c.GainReduction(), epsilon, "sample %d", i)
		wantEnv *= 1 - releaseCoef // release toward 0 dB
	}
}

// A finite attack time makes the first sample only cover the coefficient's fraction of the
// target reduction.
func TestCompressor_FiniteAttackCoefficient(t *testing.T) {
	sr := beep.SampleRate(100)
	attack := time.Second
	s := constantStreamer(1, 1.0)
	c, err := effects.NewCompressor(s, sr, testThresholdDb, testRatio, attack, instant, 0)
	assert.NoError(t, err)

	attackCoef := math.Exp(-1 / (float64(sr) * attack.Seconds()))
	buf := make([][2]float64, 1)
	n, ok := c.Stream(buf)
	assert.Equal(t, 1, n)
	assert.True(t, ok)
	assert.InDelta(t, expectedReduction*attackCoef, c.GainReduction(), epsilon)
}

// A one-sample transient with instant release stops reducing again immediately afterwards.
func TestCompressor_TransientPulse(t *testing.T) {
	sr := beep.SampleRate(100)
	data := make([][2]float64, 4)
	data[1] = [2]float64{1, 1}
	s := testtools.NewDataStreamer(data)
	c, err := effects.NewCompressor(s, sr, testThresholdDb, testRatio, time.Second, instant, 0)
	assert.NoError(t, err)

	attackCoef := math.Exp(-1 / (float64(sr) * time.Second.Seconds()))
	buf := make([][2]float64, 1)
	want := []float64{0, expectedReduction * attackCoef, 0, 0}
	for i := range want {
		n, ok := c.Stream(buf)
		assert.Equal(t, 1, n)
		assert.True(t, ok)
		assert.InDelta(t, want[i], c.GainReduction(), epsilon, "sample %d", i)
	}
}

// Output must not depend on how Stream calls are chunked.
func TestCompressor_ChunkingIndependence(t *testing.T) {
	_, data := testtools.RandomDataStreamer(2000)
	mk := func() *effects.Compressor {
		return newTestCompressor(testtools.NewDataStreamer(data))
	}

	reference := collectChunks(mk(), 2000, 2000)
	for _, chunk := range []int{1, 2, 3, 7, 64, 479} {
		got := collectChunks(mk(), 2000, chunk)
		assertSamplesClose(t, reference, got)
	}
}

// Silence passes through as silence and fully releases the envelope.
func TestCompressor_Silence(t *testing.T) {
	s := constantStreamer(64, 0)
	c := newTestCompressor(s)

	got := testtools.Collect(c)
	for i, g := range got {
		assert.Zero(t, g[0], "sample %d", i)
		assert.Zero(t, g[1], "sample %d", i)
	}
	assert.Zero(t, c.GainReduction())
}

// After compression, a run of silence recovers the envelope back to unity gain.
func TestCompressor_EnvelopeRecoversOnSilence(t *testing.T) {
	data := make([][2]float64, 20)
	for i := 0; i < 5; i++ {
		data[i] = [2]float64{1, 1}
	}
	s := testtools.NewDataStreamer(data)
	// At sr=100 a 50 ms release time constant makes the one-pole factor per sample
	// 1-exp(-0.2) ~= 0.181, so the envelope vanishes below 1e-9 within the 15 silent samples.
	c, err := effects.NewCompressor(s, beep.SampleRate(100), testThresholdDb, testRatio, instant, time.Millisecond*50, 0)
	assert.NoError(t, err)

	got := testtools.Collect(c)
	assert.InDelta(t, compressedUnity, got[4][0], epsilon)
	assert.InDelta(t, 0.0, c.GainReduction(), 1e-9)
}

// Streams shorter than any buffer, including empty ones, behave deterministically.
func TestCompressor_VeryShortStreams(t *testing.T) {
	// Single loud sample.
	one := constantStreamer(1, 1.0)
	c := newTestCompressor(one)
	buf := make([][2]float64, 16)
	n, ok := c.Stream(buf)
	assert.Equal(t, 1, n)
	assert.True(t, ok)
	assert.InDelta(t, compressedUnity, buf[0][0], epsilon)

	n, ok = c.Stream(buf)
	assert.Zero(t, n)
	assert.False(t, ok)
	assert.NoError(t, c.Err())

	// Empty stream.
	empty := testtools.NewDataStreamer(nil)
	c2 := newTestCompressor(empty)
	n, ok = c2.Stream(buf)
	assert.Zero(t, n)
	assert.False(t, ok)
	assert.Zero(t, c2.GainReduction())
}

// Stereo link: the louder channel drives one gain applied identically to both channels.
func TestCompressor_StereoLinked(t *testing.T) {
	data := [][2]float64{{1.0, 0.05}} // right alone is below threshold
	s := testtools.NewDataStreamer(data)
	c := newTestCompressor(s)

	got := testtools.Collect(c)
	gainLeft := got[0][0] / data[0][0]
	gainRight := got[0][1] / data[0][1]
	assert.InDelta(t, gainLeft, gainRight, epsilon)
	assert.InDelta(t, compressedUnity, got[0][0], epsilon)
	assert.InDelta(t, 0.05*compressedUnity, got[0][1], epsilon)
}

// Reset drops all carried-over gain reduction.
func TestCompressor_Reset(t *testing.T) {
	s := constantStreamer(4, 1.0)
	c, err := effects.NewCompressor(s, beep.SampleRate(100), testThresholdDb, testRatio, time.Second, instant, 0)
	assert.NoError(t, err)

	got := testtools.CollectNum(2, c)
	assert.NotZero(t, got[0][0])
	assert.NotZero(t, c.GainReduction())

	c.Reset()
	assert.Zero(t, c.GainReduction())

	// After reset the next loud sample reacts exactly like the first sample ever did.
	attackCoef := math.Exp(-1 / (100 * time.Second.Seconds()))
	buf := make([][2]float64, 1)
	n, ok := c.Stream(buf)
	assert.Equal(t, 1, n)
	assert.True(t, ok)
	assert.InDelta(t, expectedReduction*attackCoef, c.GainReduction(), epsilon)
}

// Changing the sample rate takes effect from the next Stream call and re-derives coefficients;
// the envelope state itself carries over continuously.
func TestCompressor_SampleRateChange(t *testing.T) {
	c, err := effects.NewCompressor(constantStreamer(4, 1.0), beep.SampleRate(100),
		testThresholdDb, testRatio, time.Second, instant, 0)
	assert.NoError(t, err)

	buf := make([][2]float64, 1)
	n, ok := c.Stream(buf)
	assert.Equal(t, 1, n)
	assert.True(t, ok)
	assert.InDelta(t, expectedReduction*math.Exp(-1.0/100), c.GainReduction(), epsilon)

	c.Reset()
	c.SampleRate = beep.SampleRate(200)
	n, ok = c.Stream(buf)
	assert.Equal(t, 1, n)
	assert.True(t, ok)
	assert.InDelta(t, expectedReduction*math.Exp(-1.0/200), c.GainReduction(), epsilon)
}

func TestCompressor_NewCompressorRejectsInvalidParams(t *testing.T) {
	valid := func() *effects.Compressor {
		return &effects.Compressor{
			Streamer:   constantStreamer(1, 0),
			SampleRate: testSR,
			Threshold:  testThresholdDb,
			Ratio:      testRatio,
			Attack:     instant,
			Release:    instant,
			MakeupGain: 0,
		}
	}

	cases := []struct {
		name   string
		mutate func(*effects.Compressor)
	}{
		{"nil streamer", func(c *effects.Compressor) { c.Streamer = nil }},
		{"zero sample rate", func(c *effects.Compressor) { c.SampleRate = 0 }},
		{"negative sample rate", func(c *effects.Compressor) { c.SampleRate = -1 }},
		{"NaN threshold", func(c *effects.Compressor) { c.Threshold = math.NaN() }},
		{"+Inf threshold", func(c *effects.Compressor) { c.Threshold = math.Inf(1) }},
		{"zero ratio", func(c *effects.Compressor) { c.Ratio = 0 }},
		{"ratio below one", func(c *effects.Compressor) { c.Ratio = 0.5 }},
		{"NaN ratio", func(c *effects.Compressor) { c.Ratio = math.NaN() }},
		{"Inf ratio", func(c *effects.Compressor) { c.Ratio = math.Inf(1) }},
		{"negative attack", func(c *effects.Compressor) { c.Attack = -time.Millisecond }},
		{"negative release", func(c *effects.Compressor) { c.Release = -time.Millisecond }},
		{"NaN makeup gain", func(c *effects.Compressor) { c.MakeupGain = math.NaN() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := valid()
			tc.mutate(c)
			got, err := effects.NewCompressor(c.Streamer, c.SampleRate, c.Threshold, c.Ratio, c.Attack, c.Release, c.MakeupGain)
			assert.Error(t, err)
			assert.Nil(t, got)
		})
	}
}

// Structs configured with invalid fields outside the constructor degrade to pass-through
// instead of producing NaNs.
func TestCompressor_InvalidRuntimeConfigIsPassThrough(t *testing.T) {
	base := func() (*effects.Compressor, [][2]float64) {
		s, data := testtools.RandomDataStreamer(32)
		return &effects.Compressor{
			Streamer:   s,
			SampleRate: testSR,
			Threshold:  testThresholdDb,
			Ratio:      testRatio,
			Attack:     instant,
			Release:    instant,
		}, data
	}

	c, data := base()
	c.SampleRate = 0
	assertSamplesClose(t, data, testtools.Collect(c))

	c, data = base()
	c.Ratio = math.NaN()
	assertSamplesClose(t, data, testtools.Collect(c))

	c, data = base()
	c.Attack = -time.Second
	assertSamplesClose(t, data, testtools.Collect(c))

	c, data = base()
	c.Threshold = math.Inf(1)
	assertSamplesClose(t, data, testtools.Collect(c))

	// A sub-unity ratio set at runtime is clamped to unity (no compression) instead of
	// expanding the signal.
	c, data = base()
	c.Ratio = 0.5
	assertSamplesClose(t, data, testtools.Collect(c))
}

// Errors from the wrapped streamer are propagated and the streamer stays drained.
func TestCompressor_PropagatesErrors(t *testing.T) {
	errBoom := errors.New("boom")
	s := testtools.NewDelayedErrorStreamer(constantStreamer(10, 1.0), 5, errBoom)
	c := newTestCompressor(s)

	got := testtools.Collect(c)
	assert.Len(t, got, 5)
	assert.Equal(t, errBoom, c.Err())

	n, ok := c.Stream(make([][2]float64, 8))
	assert.Zero(t, n)
	assert.False(t, ok)
	assert.Equal(t, errBoom, c.Err())
}

// The Streamer contract return patterns are respected.
func TestCompressor_ReturnBehaviour(t *testing.T) {
	s, _ := testtools.RandomDataStreamer(200)
	c := newTestCompressor(s)
	testtools.AssertStreamerHasCorrectReturnBehaviour(t, c, 200)
}
