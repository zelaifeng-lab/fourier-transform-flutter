"""Independent transform-pair checks at the API's pre-render output boundary.

Expected spectra follow the defining integral, delta sifting, F{1}=2*pi*delta,
and F{u}=pi*delta-i*PV(1/w); differentiation gives polynomial spectra.
No expected result is obtained from a production rule or LaTeX renderer.
"""
import time
import pytest
import sympy as s
from fastapi.testclient import TestClient
import backend as b

w = b.omega
PV = b.PV
D = s.DiracDelta
client = TestClient(b.app)

@pytest.fixture
def captured(monkeypatch):
    outputs = []
    original = b._finalize_response
    def capture(f, result, method, **kwargs):
        payload = original(f, result, method, **kwargs)
        outputs.append((s.sympify(result[2]), payload))
        return payload
    monkeypatch.setattr(b, '_finalize_response', capture)
    return outputs

def post(expression):
    start = time.monotonic()
    response = client.post('/fourier', json={'expression': expression})
    assert response.status_code == 200
    assert time.monotonic()-start < 20
    data = response.json()
    for field in ['result_latex','conditions_latex','steps_latex']:
        assert not b._bad_display(str(data[field]))
    return data

def components(expr):
    """Canonical delta coefficients by location/order; separate PV and regular parts.

    h(w) delta^(n)(w-c) = sum_k (-1)^k C(n,k) h^(k)(c)
    delta^(n-k)(w-c). This tests distributions, not point samples.
    """
    deltas, pvs, regular = {}, {}, s.S.Zero
    for term in s.Add.make_args(s.expand(expr)):
        ds = list(term.atoms(D))
        ps = list(term.atoms(PV))
        assert not (ds and ps), 'Illegal delta/PV product'
        assert len(ds) <= 1 and len(ps) <= 1
        if ds:
            atom = ds[0]
            coefficient = term/atom
            assert not coefficient.has(D,PV)
            a = s.diff(atom.args[0],w)
            c = s.simplify(-atom.args[0].subs(w,0)/a)
            assert not a.has(w)
            n = int(atom.args[1]) if len(atom.args)>1 else 0
            h = coefficient/(s.Abs(a)*a**n)
            for k in range(n+1):
                key = (c,n-k)
                value = (-1)**k*s.binomial(n,k)*s.diff(h,w,k).subs(w,c)
                deltas[key] = deltas.get(key,0)+value
        elif ps:
            atom=ps[0]
            coefficient=term/atom
            assert not coefficient.has(D,PV)
            key=s.cancel(atom.args[0])
            pvs[key]=pvs.get(key,0)+coefficient
        else:
            regular += term
    return deltas,pvs,regular

def assert_distribution(actual, expected):
    ad,ap,ar=components(actual)
    ed,ep,er=components(expected)
    for key in ad.keys() | ed.keys():
        assert s.simplify(ad.get(key,0)-ed.get(key,0)) == 0, ('delta',key,ad,ed)
    for key in ap.keys() | ep.keys():
        assert s.simplify(ap.get(key,0)-ep.get(key,0)) == 0, ('PV',key,ap,ep)
    assert s.simplify(ar-er) == 0

@pytest.mark.parametrize('expression, expected', [
    ('3*delta(2*t-6)', s.Rational(3,2)*s.exp(-3*s.I*w)),
    ('delta(-2*t+6,1)', -s.I*w*s.exp(-3*s.I*w)/4),
    ('2*exp(-3*(t-2))*u(2*t-4)', 2*s.exp(-2*s.I*w)/(3+s.I*w)),
    ('3*exp(2*t+1)*u(-2*t-4)', 3*s.exp(-3+2*s.I*w)/(2-s.I*w)),
    ('exp(-2*t^2)',s.sqrt(s.pi/2)*s.exp(-w*w/8)),
    ('exp(I*3*t)*exp(-(t+1)^2)',s.sqrt(s.pi)*s.exp(s.I*(w-3))*s.exp(-(w-3)**2/4)),
    ('1/(t^2+1)',s.pi*s.exp(-s.Abs(w))),
    ('1/(t^2+1)•1/(t^2+1)',s.pi**2*s.exp(-2*s.Abs(w))),
    ('exp(-2*t)*u(t)•exp(-3*t)*u(t)',1/((2+s.I*w)*(3+s.I*w))),
])
def test_ordinary_symbolic_equivalence(expression,expected,captured):
    data=post(expression)
    assert data['ok'], data
    assert s.simplify(captured[-1][0]-expected)==0

@pytest.mark.parametrize('expression, expected',[
    ('1',2*s.pi*D(w)),
    ('u(t)',s.pi*D(w)-s.I*PV(1/w)),
    ('u(t-2)',s.exp(-2*s.I*w)*(s.pi*D(w)-s.I*PV(1/w))),
    ('3*u(-2*t+4)',3*s.exp(-2*s.I*w)*(s.pi*D(w)+s.I*PV(1/w))),
    ('sign(t-2)',-2*s.I*s.exp(-2*s.I*w)*PV(1/w)),
    ('sin(3*t+2)',s.pi/s.I*(s.exp(2*s.I)*D(w-3)-s.exp(-2*s.I)*D(w+3))),
    ('2*t^2+3*t+1',-4*s.pi*D(w,2)+6*s.pi*s.I*D(w,1)+2*s.pi*D(w)),
    ('t*u(t)',s.I*s.pi*D(w,1)-PV(1/w**2)),
    ('t^2*u(t)',-s.pi*D(w,2)+2*s.I*PV(1/w**3)),
    ('(t-2)*u(t-2)',s.exp(-2*s.I*w)*(s.I*s.pi*D(w,1)-PV(1/w**2))),
    ('u(t)•u(t)',s.I*s.pi*D(w,1)-PV(1/w**2)),
    ('delta(t-1)•sin(t)',s.exp(-s.I*w)*s.pi/s.I*(D(w-1)-D(w+1))),
    ('u(t)•exp(-t)*u(t)',(s.pi*D(w)-s.I*PV(1/w))/(1+s.I*w)),
])
def test_distribution_components(expression,expected,captured):
    data=post(expression)
    assert data['ok'], data
    assert_distribution(captured[-1][0],expected)

@pytest.mark.parametrize('expression,condition',[
    ('exp(-a*t)*u(t)','a>0'),
    ('exp(-a*t^2)','a>0'),
    ('exp(-a*abs(t))','a>0'),
    ('delta(a*t-2)',r'a\ne0'),
    ('exp(-a*t)*u(t)•exp(-2*t)*u(t)','a>0'),
])
def test_parameter_conditions(expression,condition):
    data=post(expression)
    assert data['ok'],data
    assert condition in ''.join(data['conditions_latex'].split())

@pytest.mark.parametrize('expression',[
    'cos(t)•cos(t)', '1•1', 'u(t)•u(-t)', 'exp(t)*u(t)',
    'exp(-t)*u(-t)',
])
def test_undefined_or_divergent_is_not_success(expression):
    data=post(expression)
    assert not data['ok']
    assert data['error']
    assert data['form'] in ['integral_form','divergent']
    assert r'\delta^{2}' not in data['result_latex']

@pytest.mark.parametrize('expression',[
    '__import__("os")', 't.__class__', 'open(t)', 'sin(t)[0]',
    'lambda:t', 't if t else 1', 'Symbol(t)', 'foo(t)', 't;1',
    'frac(1,t', 'sin(t,)', '1/0', 't**1001', '('*33+'t'+')'*33,
])
def test_restricted_parser_rejects_python_syntax(expression):
    data=post(expression)
    assert not data['ok']
    assert data['form']=='error'
    assert data['method']=='input_validation'
    assert data['error'].startswith('Parser error')

@pytest.mark.parametrize('expression',[
    'frac(1,frac(t^2+1,2))','2sin（t）','frac(1,(t+1)(t+2))',
    'exp(-alpha*t)*u(t)','delta（t-1）•sin（t）','3t',
])
def test_supported_surface_notation(expression):
    assert post(expression)['ok']

@pytest.mark.parametrize('expression,method',[
    ('exp(-2*t^2)','known_pair'),('u(t)','distribution_rule'),
    ('exp(-(t+1)^2)','property_rule'),('u(t)•u(t)','convolution_rule'),
])
def test_method_reports_executed_path(expression,method):
    data=post(expression)
    assert data['ok']
    assert data['method']==method

def test_unevaluated_integral_remains_unsolved(monkeypatch):
    monkeypatch.setattr(b.Integral,'doit',lambda self,**kwargs:self)
    data=post('log(t^2+1)')
    assert not data['ok']
    assert data['form']=='integral_form'
    assert data['method']=='direct_integral'
    assert r'\int' in data['result_latex']

@pytest.mark.parametrize('bad',[
    s.Piecewise((1,w>0),(2,True)), s.arg(w), s.polar_lift(w),
    s.meijerg([],[],[0],[],w),
])
def test_complete_candidate_rejected_without_deleting_math(bad,monkeypatch):
    monkeypatch.setattr(b,'_derive_with_properties',lambda f:('closed_form',True,bad,[s.latex(bad)],'',None))
    data=post('t')
    assert not data['ok']
    assert data['form']=='integral_form'
    assert r'\int' in data['result_latex']

@pytest.mark.parametrize('where',['steps','conditions'])
def test_internal_text_rejects_candidate(where,monkeypatch):
    steps=['debug internal result'] if where=='steps' else []
    conditions='matcher internals' if where=='conditions' else ''
    monkeypatch.setattr(b,'_derive_with_properties',lambda f:('closed_form',True,s.S.One,steps,conditions,None))
    assert not post('t')['ok']

def test_component_oracle_detects_wrong_order_coefficient_and_pv():
    expected=s.pi*D(w-2,1)+PV(1/w)
    for wrong in [s.pi*D(w-3,1)+PV(1/w),s.pi*D(w-2)+PV(1/w),
                  2*s.pi*D(w-2,1)+PV(1/w),s.pi*D(w-2,1)-PV(1/w)]:
        with pytest.raises(AssertionError): assert_distribution(wrong,expected)


def test_R08_independent_residues(captured):
    # 2t/(3t^2+4t-1): simple-pole residue N(c)/D'(c).
    centers=[(-2-s.sqrt(7))/3,(-2+s.sqrt(7))/3]
    expected=sum(-s.I*s.pi*(2*c/(6*c+4))*s.exp(-s.I*w*c)*s.sign(w) for c in centers)
    data=post('frac(2*t,3*t^2+4*t-1)')
    assert data['ok']
    assert s.simplify(captured[-1][0]-expected)==0


def test_causal_step_convolution_shift_and_coefficients(captured):
    data=post('2*u(t-1)•3*u(t-2)')
    assert data['ok']
    expected=6*s.exp(-3*s.I*w)*(s.I*s.pi*D(w,1)-PV(1/w**2))
    assert_distribution(captured[-1][0],expected)


def test_conditions_not_dropped_by_rational_linearity():
    data=post('1/(t+1)+1/(t^2+a^2)')
    # A zero quadratic parameter creates a different singular distribution.
    assert not data['ok'] or data['conditions_latex']


@pytest.mark.parametrize('expression',['3*exp(-t^2)','u(t)•u(t)','2*sin(t)+cos(t)','delta(t-2)•u(t)'])
def test_teaching_finishes_with_actual_result(expression):
    data=post(expression)
    assert data['ok']
    assert data['steps_latex'][-2]==r'\textbf{Final Result}'
    assert data['steps_latex'][-1]==r'X(\omega)='+data['result_latex']
