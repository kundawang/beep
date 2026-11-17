package effects

import (
	"math"
	"sort"

	"github.com/gopxl/beep/v2"
)

// PanPoint is a single keyframe of a pan automation curve. Position is the index of the sample
// (relative to the start of the wrapped Streamer) where Pan must be reached. Pan ranges from -1
// (hard left) to +1 (hard right); 0 is the center.
type PanPoint struct {
	Position int
	Pan      float64
}

// NewPanAutomation returns a Streamer which pans the wrapped Streamer according to the given
// keyframe curve. The pan value at any sample position is obtained by linearly interpolating
// between the two surrounding keyframes; positions before the first keyframe keep the first
// keyframe's pan and positions after the last keyframe keep the last one's.
//
// The left and right channel gains follow a constant-power (equal-power) pan law:
//
//	gainL = cos(theta), gainR = sin(theta), with theta = (pan + 1) * pi / 4
//
// so that gainL^2 + gainR^2 == 1 for every pan value and the perceived loudness does not dip in
// the middle of the pan sweep. At pan -1 only the left channel remains, at pan +1 only the right
// channel, and at pan 0 both channels are scaled by sqrt(2)/2.
//
// The points are normalized on construction:
//
//   - Points may be given in any order; they are sorted by Position in ascending order.
//   - Multiple points sharing the same Position are merged, keeping the pan value of the last
//     point at that position in the input slice.
//   - Negative Positions are clamped to 0.
//   - Pan values outside [-1, +1] are clamped to that range; a NaN pan becomes 0 (center).
//   - An empty points slice disables the effect: the samples are streamed unchanged.
//
// Mono input (both channels carrying the same value) needs no special treatment and is panned
// exactly like stereo input: at the center position the mono signal is split at -3 dB per side,
// preserving the total energy across the whole sweep.
//
// If the wrapped Streamer is a beep.StreamSeeker, the returned Streamer is one as well: seeking
// restores both the source position and the pan curve state at that position. Errors of the
// wrapped Streamer are propagated through Err.
func NewPanAutomation(s beep.Streamer, points []PanPoint) beep.Streamer {
	pa := &PanAutomation{
		streamer: s,
		points:   normalizePanPoints(points),
	}
	if ss, ok := s.(beep.StreamSeeker); ok {
		return &panAutomationSeeker{PanAutomation: pa, seeker: ss}
	}
	return pa
}

// PanAutomation applies a keyframe-driven, constant-power pan automation to its Streamer. Use
// NewPanAutomation to construct one.
type PanAutomation struct {
	streamer beep.Streamer
	points   []PanPoint
	pos      int
	seg      int
}

// panAutomationSeeker extends PanAutomation with the StreamSeeker interface for seekable sources.
type panAutomationSeeker struct {
	*PanAutomation
	seeker beep.StreamSeeker
}

// normalizePanPoints returns a sorted, position-deduplicated and value-clamped copy of points.
func normalizePanPoints(points []PanPoint) []PanPoint {
	if len(points) == 0 {
		return nil
	}
	normalized := make([]PanPoint, len(points))
	copy(normalized, points)
	for i := range normalized {
		if normalized[i].Position < 0 {
			normalized[i].Position = 0
		}
		normalized[i].Pan = clampPan(normalized[i].Pan)
	}

	sort.SliceStable(normalized, func(i, j int) bool {
		return normalized[i].Position < normalized[j].Position
	})

	// Merge points at the same position, keeping the last one in the (stable) input order.
	deduped := make([]PanPoint, 0, len(normalized))
	for _, p := range normalized {
		if len(deduped) > 0 && deduped[len(deduped)-1].Position == p.Position {
			deduped[len(deduped)-1] = p
			continue
		}
		deduped = append(deduped, p)
	}
	return deduped
}

func clampPan(pan float64) float64 {
	if math.IsNaN(pan) {
		return 0
	}
	if pan < -1 {
		return -1
	}
	if pan > 1 {
		return 1
	}
	return pan
}

// Stream streams the wrapped Streamer with the constant-power pan curve applied sample by sample.
func (p *PanAutomation) Stream(samples [][2]float64) (n int, ok bool) {
	n, ok = p.streamer.Stream(samples)
	if len(p.points) == 0 {
		p.pos += n
		return n, ok
	}
	for i := 0; i < n; i++ {
		pan := p.panAt(p.pos + i)
		theta := (pan + 1) * math.Pi / 4
		gainL, gainR := math.Cos(theta), math.Sin(theta)
		samples[i][0] *= gainL
		samples[i][1] *= gainR
	}
	p.pos += n
	return n, ok
}

// panAt returns the (interpolated) pan value at absolute sample position pos.
func (p *PanAutomation) panAt(pos int) float64 {
	// Advance the cached segment while the position lies beyond its end. Segments are
	// traversed monotonically during streaming; Seek resets the cache.
	for p.seg < len(p.points)-1 && pos >= p.points[p.seg+1].Position {
		p.seg++
	}
	if p.seg >= len(p.points)-1 {
		return p.points[len(p.points)-1].Pan
	}
	a, b := p.points[p.seg], p.points[p.seg+1]
	if pos <= a.Position {
		return a.Pan
	}
	frac := float64(pos-a.Position) / float64(b.Position-a.Position)
	return a.Pan + (b.Pan-a.Pan)*frac
}

// resetTo moves the curve state to the given absolute sample position.
func (p *PanAutomation) resetTo(pos int) {
	p.pos = pos
	if len(p.points) == 0 {
		p.seg = 0
		return
	}
	// Locate the first segment ending at or after pos with a binary search; panAt then walks
	// the cache forward if needed.
	p.seg = sort.Search(len(p.points)-1, func(i int) bool {
		return pos < p.points[i+1].Position
	})
}

// Err propagates the wrapped Streamer's errors.
func (p *PanAutomation) Err() error {
	return p.streamer.Err()
}

// Len propagates the wrapped StreamSeeker's length.
func (p *panAutomationSeeker) Len() int {
	return p.seeker.Len()
}

// Position propagates the wrapped StreamSeeker's position.
func (p *panAutomationSeeker) Position() int {
	return p.seeker.Position()
}

// Seek sets the position of the wrapped StreamSeeker and restores the pan curve state at the
// target sample position. A seek error from the wrapped StreamSeeker is returned unchanged and
// leaves the curve state untouched.
func (p *panAutomationSeeker) Seek(pos int) error {
	if err := p.seeker.Seek(pos); err != nil {
		return err
	}
	p.resetTo(pos)
	return nil
}
