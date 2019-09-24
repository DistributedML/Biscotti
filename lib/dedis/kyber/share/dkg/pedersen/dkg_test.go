package dkg

import (
	"crypto/rand"
	"testing"

	"github.com/dedis/kyber"
	"github.com/dedis/kyber/group/edwards25519"
	"github.com/dedis/kyber/share"
	vss "github.com/dedis/kyber/share/vss/pedersen"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var suite = edwards25519.NewBlakeSHA256Ed25519()

var nbParticipants = 7

var partPubs []kyber.Point
var partSec []kyber.Scalar

var dkgs []*DistKeyGenerator
var dkgsReNew []*DistKeyGenerator

func init() {
	partPubs = make([]kyber.Point, nbParticipants)
	partSec = make([]kyber.Scalar, nbParticipants)
	for i := 0; i < nbParticipants; i++ {
		sec, pub := genPair()
		partPubs[i] = pub
		partSec[i] = sec
	}
	dkgs = dkgGen()
}

func TestDKGNewDistKeyGenerator(t *testing.T) {
	long := partSec[0]
	dkg, err := NewDistKeyGenerator(suite, long, partPubs, nbParticipants/2+1)
	assert.Nil(t, err)
	assert.NotNil(t, dkg.dealer)

	sec, _ := genPair()
	_, err = NewDistKeyGenerator(suite, sec, partPubs, nbParticipants/2+1)
	assert.Error(t, err)
}

func TestDKGDeal(t *testing.T) {
	dkg := dkgs[0]

	dks, err := dkg.DistKeyShare()
	assert.Error(t, err)
	assert.Nil(t, dks)

	deals, err := dkg.Deals()
	require.Nil(t, err)
	assert.Len(t, deals, nbParticipants-1)

	for i := range deals {
		assert.NotNil(t, deals[i])
		assert.Equal(t, uint32(0), deals[i].Index)
	}

	v, ok := dkg.verifiers[dkg.index]
	assert.True(t, ok)
	assert.NotNil(t, v)
}

func TestDKGProcessDeal(t *testing.T) {
	dkgs = dkgGen()
	dkg := dkgs[0]
	deals, err := dkg.Deals()
	require.Nil(t, err)

	rec := dkgs[1]
	deal := deals[1]
	assert.Equal(t, int(deal.Index), 0)
	assert.Equal(t, uint32(1), rec.index)

	// verifier don't find itself
	goodP := rec.participants
	rec.participants = make([]kyber.Point, 0)
	resp, err := rec.ProcessDeal(deal)
	assert.Nil(t, resp)
	assert.Error(t, err)
	rec.participants = goodP

	// good deal
	resp, err = rec.ProcessDeal(deal)
	assert.NotNil(t, resp)
	assert.Equal(t, vss.StatusApproval, resp.Response.Status)
	assert.Nil(t, err)
	_, ok := rec.verifiers[deal.Index]
	require.True(t, ok)
	assert.Equal(t, uint32(0), resp.Index)

	// duplicate
	resp, err = rec.ProcessDeal(deal)
	assert.Nil(t, resp)
	assert.Error(t, err)

	// wrong index
	goodIdx := deal.Index
	deal.Index = uint32(nbParticipants + 1)
	resp, err = rec.ProcessDeal(deal)
	assert.Nil(t, resp)
	assert.Error(t, err)
	deal.Index = goodIdx

	// wrong deal
	goodSig := deal.Deal.Signature
	deal.Deal.Signature = randomBytes(len(deal.Deal.Signature))
	resp, err = rec.ProcessDeal(deal)
	assert.Nil(t, resp)
	assert.Error(t, err)
	deal.Deal.Signature = goodSig

}

func TestDKGProcessResponse(t *testing.T) {
	// first peer generates wrong deal
	// second peer processes it and returns a complaint
	// first peer process the complaint

	dkgs = dkgGen()
	dkg := dkgs[0]
	idxRec := 1
	rec := dkgs[idxRec]
	deal, err := dkg.dealer.PlaintextDeal(idxRec)
	require.Nil(t, err)

	// give a wrong deal
	goodSecret := deal.SecShare.V
	deal.SecShare.V = suite.Scalar().Zero()
	dd, err := dkg.Deals()
	encD := dd[idxRec]
	require.Nil(t, err)
	resp, err := rec.ProcessDeal(encD)
	assert.Nil(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, vss.StatusComplaint, resp.Response.Status)
	deal.SecShare.V = goodSecret
	dd, _ = dkg.Deals()
	encD = dd[idxRec]

	// no verifier tied to Response
	v, ok := dkg.verifiers[0]
	require.NotNil(t, v)
	require.True(t, ok)
	require.NotNil(t, v)
	delete(dkg.verifiers, 0)
	j, err := dkg.ProcessResponse(resp)
	assert.Nil(t, j)
	assert.NotNil(t, err)
	dkg.verifiers[0] = v

	// invalid response
	goodSig := resp.Response.Signature
	resp.Response.Signature = randomBytes(len(goodSig))
	j, err = dkg.ProcessResponse(resp)
	assert.Nil(t, j)
	assert.Error(t, err)
	resp.Response.Signature = goodSig

	// valid complaint from our deal
	j, err = dkg.ProcessResponse(resp)
	assert.NotNil(t, j)
	assert.Nil(t, err)

	// valid complaint from another deal from another peer
	dkg2 := dkgs[2]
	require.Nil(t, err)
	// fake a wrong deal
	// deal20, err := dkg2.dealer.PlaintextDeal(0)
	// require.Nil(t, err)
	deal21, err := dkg2.dealer.PlaintextDeal(1)
	require.Nil(t, err)
	goodRnd21 := deal21.SecShare.V
	deal21.SecShare.V = suite.Scalar().Zero()
	deals2, err := dkg2.Deals()
	require.Nil(t, err)

	resp12, err := rec.ProcessDeal(deals2[idxRec])
	assert.NotNil(t, resp)
	assert.Equal(t, vss.StatusComplaint, resp12.Response.Status)

	deal21.SecShare.V = goodRnd21
	deals2, err = dkg2.Deals()
	require.Nil(t, err)

	// give it to the first peer
	// process dealer 2's deal
	r, err := dkg.ProcessDeal(deals2[0])
	assert.Nil(t, err)
	assert.NotNil(t, r)

	// process response from peer 1
	j, err = dkg.ProcessResponse(resp12)
	assert.Nil(t, j)
	assert.Nil(t, err)

	// Justification part:
	// give the complaint to the dealer
	j, err = dkg2.ProcessResponse(resp12)
	assert.Nil(t, err)
	assert.NotNil(t, j)

	// hack because all is local, and resp has been modified locally by dkg2's
	// dealer, the status has became "justified"
	resp12.Response.Status = vss.StatusComplaint
	err = dkg.ProcessJustification(j)
	assert.Nil(t, err)

	// remove verifiers
	v = dkg.verifiers[j.Index]
	delete(dkg.verifiers, j.Index)
	err = dkg.ProcessJustification(j)
	assert.Error(t, err)
	dkg.verifiers[j.Index] = v

}

func TestSetTimeout(t *testing.T) {
	dkgs = dkgGen()
	// full secret sharing exchange
	// 1. broadcast deals
	resps := make([]*Response, 0, nbParticipants*nbParticipants)
	for _, dkg := range dkgs {
		deals, err := dkg.Deals()
		require.Nil(t, err)
		for i, d := range deals {
			resp, err := dkgs[i].ProcessDeal(d)
			require.Nil(t, err)
			require.Equal(t, vss.StatusApproval, resp.Response.Status)
			resps = append(resps, resp)
		}
	}

	// 2. Broadcast responses
	for _, resp := range resps {
		for _, dkg := range dkgs {
			if !dkg.verifiers[resp.Index].EnoughApprovals() {
				// ignore messages about ourself
				if resp.Response.Index == dkg.index {
					continue
				}
				j, err := dkg.ProcessResponse(resp)
				require.Nil(t, err)
				require.Nil(t, j)
			}
		}
	}

	// 3. make sure everyone has the same QUAL set
	for _, dkg := range dkgs {
		for _, dkg2 := range dkgs {
			require.False(t, dkg.isInQUAL(dkg2.index))
		}
	}

	for _, dkg := range dkgs {
		dkg.SetTimeout()
	}

	for _, dkg := range dkgs {
		for _, dkg2 := range dkgs {
			require.True(t, dkg.isInQUAL(dkg2.index))
		}
	}

}

func TestDistKeyShare(t *testing.T) {
	fullExchange(t)

	for _, dkg := range dkgs {
		assert.True(t, dkg.Certified())
	}
	// verify integrity of shares etc
	dkss := make([]*DistKeyShare, nbParticipants)
	var poly *share.PriPoly
	for i, dkg := range dkgs {
		dks, err := dkg.DistKeyShare()
		require.Nil(t, err)
		require.NotNil(t, dks)
		require.NotNil(t, dks.PrivatePoly)
		dkss[i] = dks
		assert.Equal(t, dkg.index, uint32(dks.Share.I))

		pripoly := share.CoefficientsToPriPoly(suite, dks.PrivatePoly)
		if poly == nil {
			poly = pripoly
			continue
		}
		poly, err = poly.Add(pripoly)
		require.NoError(t, err)
	}

	shares := make([]*share.PriShare, nbParticipants)
	for i, dks := range dkss {
		assert.True(t, checkDks(dks, dkss[0]), "dist key share not equal %d vs %d", dks.Share.I, 0)
		shares[i] = dks.Share
	}

	secret, err := share.RecoverSecret(suite, shares, nbParticipants, nbParticipants)
	assert.Nil(t, err)

	secretCoeffs := poly.Coefficients()
	require.Equal(t, secret.String(), secretCoeffs[0].String())

	commitSecret := suite.Point().Mul(secret, nil)
	assert.Equal(t, dkss[0].Public().String(), commitSecret.String())
}

func dkgGen() []*DistKeyGenerator {
	dkgs := make([]*DistKeyGenerator, nbParticipants)
	for i := 0; i < nbParticipants; i++ {
		dkg, err := NewDistKeyGenerator(suite, partSec[i], partPubs, nbParticipants/2+1)
		if err != nil {
			panic(err)
		}
		dkgs[i] = dkg
	}
	return dkgs
}

func TestDistKeyShareReNew(t *testing.T) {
	fullExchange(t)
	fullExchangeWithRenewal(t)

	dkss := make([]*DistKeyShare, nbParticipants)
	for i, dkg := range dkgs {
		dks, _ := dkg.DistKeyShare()
		dksReNew, _ := dkgsReNew[i].DistKeyShare()
		dkss[i], _ = dks.Renew(dkg.suite, dksReNew)
	}

	shares := make([]*share.PriShare, nbParticipants)
	for i, dks := range dkss {
		shares[i] = dks.Share
	}
	secret, err := share.RecoverSecret(suite, shares, nbParticipants, nbParticipants) // SUMf_j(0),j:0->#participants-
	assert.Nil(t, err)

	commitSecret := suite.Point().Mul(secret, nil)
	assert.Equal(t, dkss[0].Public().String(), commitSecret.String())
}

func TestReNewDistKeyGenerator(t *testing.T) {
	long := partSec[0]
	dkgNew, err := NewDistKeyGeneratorWithoutSecret(suite, long, partPubs, nbParticipants/2+1)
	assert.Nil(t, err)
	assert.NotNil(t, dkgNew.dealer)

}

func TestDistKeyShare_Renew(t *testing.T) {
	fullExchange(t)
	fullExchangeWithRenewal(t)
	dkg := dkgs[2]
	dks, _ := dkg.DistKeyShare()

	//Check when they don't have the same index
	dkg1 := dkgsReNew[1]
	dks1, _ := dkg1.DistKeyShare()
	newDsk1, err := dks.Renew(dkg.suite, dks1)
	assert.Nil(t, newDsk1)
	assert.Error(t, err)

	//Check the last coeff is not 0 in g(x)
	dkg3 := dkgs[1]
	dks3, _ := dkg3.DistKeyShare()
	newDsk3, err3 := dks.Renew(dkg.suite, dks3)
	assert.Nil(t, newDsk3)
	assert.Error(t, err3)

	//Finally, check whether it works
	dkg2 := dkgsReNew[2]
	dks2, _ := dkg2.DistKeyShare()
	newDks, err := dks.Renew(dkg.suite, dks2)
	assert.Nil(t, err)
	assert.NotNil(t, newDks)

}

func TestReNewDistKeyShare(t *testing.T) {
	fullExchange(t)
	fullExchangeWithRenewal(t)

	for _, dkg := range dkgs {
		assert.True(t, dkg.Certified())

	}
	// verify integrity of shares etc
	formerDks, _ := dkgs[0].DistKeyShare()
	dkss := make([]*DistKeyShare, nbParticipants)
	for i, dkg := range dkgs {
		dks, err := dkg.DistKeyShare()

		require.Nil(t, err)
		require.NotNil(t, dks)
		dksNew, _ := dkgsReNew[i].DistKeyShare()
		dkss[i], _ = dks.Renew(dkg.suite, dksNew) //Renewal
		assert.Equal(t, dkg.index, uint32(dks.Share.I))
	}

	shares := make([]*share.PriShare, nbParticipants)
	for i, dks := range dkss {
		assert.True(t, checkDks(dks, dkss[0]), "dist key share not equal %d vs %d", dks.Share.I, 0)
		shares[i] = dks.Share
	}

	secret, err := share.RecoverSecret(suite, shares, nbParticipants, nbParticipants)
	assert.Nil(t, err)

	//Check is sum(f)*G == sum(F)
	//a0*G
	commitSecret := suite.Point().Mul(secret, nil)
	assert.Equal(t, formerDks.Public().String(), commitSecret.String())
}

func reNewDkgGen() []*DistKeyGenerator {
	dkgsNew := make([]*DistKeyGenerator, nbParticipants)
	for i := 0; i < nbParticipants; i++ {
		dkgNew, err := NewDistKeyGeneratorWithoutSecret(suite, partSec[i], partPubs, nbParticipants/2+1)
		if err != nil {
			panic(err)
		}
		dkgsNew[i] = dkgNew
	}
	return dkgsNew
}

func genPair() (kyber.Scalar, kyber.Point) {
	sc := suite.Scalar().Pick(suite.RandomStream())
	return sc, suite.Point().Mul(sc, nil)
}

func randomBytes(n int) []byte {
	var buff = make([]byte, n)
	_, _ = rand.Read(buff[:])
	return buff
}
func checkDks(dks1, dks2 *DistKeyShare) bool {
	if len(dks1.Commits) != len(dks2.Commits) {
		return false
	}
	for i, p := range dks1.Commits {
		if !p.Equal(dks2.Commits[i]) {
			return false
		}
	}
	return true
}

func fullExchange(t *testing.T) {
	dkgs = dkgGen()
	// full secret sharing exchange
	// 1. broadcast deals
	resps := make([]*Response, 0, nbParticipants*nbParticipants)
	for _, dkg := range dkgs {
		deals, err := dkg.Deals()
		require.Nil(t, err)
		for i, d := range deals {
			resp, err := dkgs[i].ProcessDeal(d)
			require.Nil(t, err)
			require.Equal(t, vss.StatusApproval, resp.Response.Status)
			resps = append(resps, resp)
		}
	}
	// 2. Broadcast responses
	for _, resp := range resps {
		for _, dkg := range dkgs {
			// Ignore messages about ourselves
			if resp.Response.Index == dkg.index {
				continue
			}
			j, err := dkg.ProcessResponse(resp)
			require.Nil(t, err)
			require.Nil(t, j)
		}
	}

	// 3. make sure everyone has the same QUAL set
	for _, dkg := range dkgs {
		for _, dkg2 := range dkgs {
			require.True(t, dkg.isInQUAL(dkg2.index))
		}
	}

}

func fullExchangeWithRenewal(t *testing.T) {
	dkgsReNew = reNewDkgGen()
	// full secret sharing exchange
	// 1. broadcast deals
	resps := make([]*Response, 0, nbParticipants*nbParticipants)
	for _, dkg := range dkgsReNew {
		deals, _ := dkg.Deals()
		for i, d := range deals {
			resp, _ := dkgsReNew[i].ProcessDeal(d)
			resps = append(resps, resp)
		}
	}
	// 2. Broadcast responses
	for _, resp := range resps {
		for _, dkg := range dkgsReNew {
			// Ignore messages about ourselves
			if resp.Response.Index == dkg.index {
				continue
			}
			j, _ := dkg.ProcessResponse(resp)
			//require.Nil(t, err)
			require.Nil(t, j)
		}
	}
}
