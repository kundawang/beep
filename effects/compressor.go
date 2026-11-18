package effects

import (
	"errors"
	"math"
	"time"

	"github.com/gopxl/beep/v2"
)

// Compressor is a streaming feed-forward dynamic range compressor.
//
// The compressor lowers the volume of samples whose level rises above the Threshold. The amount
// of lowering is determined by the Ratio: a ratio of 4 means that a signal 4 dB above the
// threshold ends up only 1 dB above it. The Attack and Release durations control how quickly the
// compressor reacts to a loud signal and how quickly it recovers once the signal gets quiet
// again. A static MakeupGain is applied on top of the dynamic reduction to compensate for the
// lost volume.
//
// The two channels are always linked: a single detector level of max(|left|, |right|) drives one
// gain envelope and the same gain is applied to both channels, so the stereo image never shifts
// to one side.
//
// All level parameters use decibels. The envelope is advanced one sample at a time with
// coefficients derived from SampleRate, so the produced output is independent of the chunk sizes
// in which Stream is called. The compressor never clips or clamps samples; with a zero
// MakeupGain the applied gain is always at most 1.0, so normal samples in the [-1, 1] range
// cannot be pushed outside of it.
//
// The zero value is a valid, fully transparent Streamer: as long as SampleRate is 0 (or any
// parameter is out of range) Stream passes the wrapped samples through unchanged. For a
// validated compressor use the NewCompressor constructor.
type Compressor struct {
	Streamer beep.Streamer

	// SampleRate is the sample rate of the wrapped Streamer. It is used to translate the
	// Attack and Release durations into per-sample envelope coefficients. If it is zero or
	// negative the compressor is disabled and acts as a pure pass-through.
	SampleRate beep.SampleRate

	// Threshold is the level in decibels above which compression starts, e.g. -24.0. Signals
	// at or below the threshold are left untouched (apart from the MakeupGain).
	Threshold float64

	// Ratio is the compression ratio expressed as x:1. A value of 4 means 4:1 compression.
	// Values below 1 are invalid; a value of 1 means no compression at all.
	Ratio float64

	// Attack is the time constant of the envelope while gain reduction is increasing. A zero
	// duration makes the compression engage instantly (within one sample).
	Attack time.Duration

	// Release is the time constant of the envelope while gain reduction is decreasing. A zero
	// duration makes the compression disengage instantly (within one sample).
	Release time.Duration

	// MakeupGain is a static gain in decibels applied to the signal after compression.
	MakeupGain float64

	// gainReductionDb is the current state of the envelope in decibels. It is always <= 0,
	// where 0 means that no reduction is currently applied.
	gainReductionDb float64
}

// NewCompressor returns a Compressor wrapping s. It validates the parameters and returns an error
// instead of constructing an effect with unusable settings:
//
//   - s must not be nil,
//   - sr must be positive,
//   - threshold and makeupGain must be finite numbers,
//   - ratio must be at least 1,
//   - attack and release must not be negative.
//
// A zero attack or release is valid and means an instant response.
func NewCompressor(s beep.Streamer, sr beep.SampleRate, threshold, ratio float64, attack, release time.Duration, makeupGain float64) (*Compressor, error) {
	c := &Compressor{
		Streamer:   s,
		SampleRate: sr,
		Threshold:  threshold,
		Ratio:      ratio,
		Attack:     attack,
		Release:    release,
		MakeupGain: makeupGain,
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	return c, nil
}

var (
	errCompressorNilStreamer = errors.New("effects: compressor requires a non-nil streamer")
	errCompressorSampleRate  = errors.New("effects: compressor requires a positive sample rate")
	errCompressorThreshold   = errors.New("effects: compressor threshold must be a finite number")
	errCompressorRatio       = errors.New("effects: compressor ratio must be a finite number >= 1")
	errCompressorAttack      = errors.New("effects: compressor attack must not be negative")
	errCompressorRelease     = errors.New("effects: compressor release must not be negative")
	errCompressorMakeupGain  = errors.New("effects: compressor makeup gain must be a finite number")
)

func (c *Compressor) validate() error {
	switch {
	case c.Streamer == nil:
		return errCompressorNilStreamer
	case c.SampleRate <= 0:
		return errCompressorSampleRate
	case !isFinite(c.Threshold):
		return errCompressorThreshold
	case !isFinite(c.Ratio) || c.Ratio < 1:
		return errCompressorRatio
	case c.Attack < 0:
		return errCompressorAttack
	case c.Release < 0:
		return errCompressorRelease
	case !isFinite(c.MakeupGain):
		return errCompressorMakeupGain
	default:
		return nil
	}
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// Stream streams the wrapped Streamer with dynamic compression applied.
func (c *Compressor) Stream(samples [][2]float64) (n int, ok bool) {
	n, ok = c.Streamer.Stream(samples)
	if n == 0 {
		// Drained or errored: there is nothing to process and the envelope state is preserved.
		return n, ok
	}

	// Defensive path for structs configured outside of NewCompressor: any invalid field makes
	// the effect fall back to deterministic pass-through rather than emitting NaNs.
	sr, thresholdDb, ratio, attack, release, makeupDb, enabled := c.effectiveParams()
	if !enabled {
		return n, ok
	}

	thresholdLin := math.Pow(10, thresholdDb/20)
	slope := 0.0
	if ratio > 1 {
		slope = 1 - 1/ratio
	}
	attackCoef := envelopeCoefficient(sr, attack)
	releaseCoef := envelopeCoefficient(sr, release)

	env := c.gainReductionDb
	for i := 0; i < n; i++ {
		// Stereo-linked detector: the louder channel controls the gain for both channels.
		level := math.Abs(samples[i][0])
		if right := math.Abs(samples[i][1]); right > level {
			level = right
		}

		// Target gain reduction in dB. Silence and anything at or below the threshold produces
		// a target of zero, which also avoids math.Log10(0).
		target := 0.0
		if level > thresholdLin {
			target = (thresholdDb - 20*math.Log10(level)) * slope
		}

		// One-pole envelope in the dB domain. Attack while reducing, release while recovering.
		if target < env {
			env += (target - env) * attackCoef
		} else {
			env += (target - env) * releaseCoef
		}

		gain := math.Pow(10, (env+makeupDb)/20)
		samples[i][0] *= gain
		samples[i][1] *= gain
	}
	c.gainReductionDb = env

	return n, ok
}

// effectiveParams returns sanitized parameters. enabled is false when the compressor cannot run
// (invalid configuration), in which case Stream must pass samples through untouched.
func (c *Compressor) effectiveParams() (sr beep.SampleRate, thresholdDb, ratio float64, attack, release time.Duration, makeupDb float64, enabled bool) {
	if c.SampleRate <= 0 ||
		!isFinite(c.Threshold) ||
		!isFinite(c.Ratio) ||
		!isFinite(c.MakeupGain) ||
		c.Attack < 0 ||
		c.Release < 0 {
		return 0, 0, 0, 0, 0, 0, false
	}

	ratio = c.Ratio
	if ratio < 1 {
		// Ratio below 1 would expand instead of compressing; treat unity as the safe boundary.
		ratio = 1
	}
	return c.SampleRate, c.Threshold, ratio, c.Attack, c.Release, c.MakeupGain, true
}

// envelopeCoefficient converts a time constant into the coefficient of a one-pole filter
// running at sr. After the given duration the envelope has covered about 63% of the distance to
// its target. A non-positive duration yields 1, i.e. the envelope reaches its target instantly.
func envelopeCoefficient(sr beep.SampleRate, t time.Duration) float64 {
	if t <= 0 {
		return 1
	}
	return math.Exp(-1 / (float64(sr) * t.Seconds()))
}

// Reset restores the envelope to unity gain (no reduction). This is useful when reusing the
// Compressor for an unrelated stream, where the gain reduction carried over from the previous
// stream would otherwise be audible.
func (c *Compressor) Reset() {
	c.gainReductionDb = 0
}

// GainReduction returns the gain reduction currently applied by the envelope, in decibels. The
// value is always less than or equal to zero; zero means the signal is currently uncompressed.
// The MakeupGain is not included.
func (c *Compressor) GainReduction() float64 {
	return c.gainReductionDb
}

// Err propagates the wrapped Streamer's errors.
func (c *Compressor) Err() error {
	return c.Streamer.Err()
}
