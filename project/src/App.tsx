import { useEffect, useMemo, useState, type CSSProperties, type FormEvent } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  Bell,
  Check,
  CheckCircle2,
  ChevronRight,
  Clapperboard,
  Eye,
  EyeOff,
  Film,
  Github,
  Home,
  LockKeyhole,
  Mail,
  Menu,
  Moon,
  Plus,
  Search,
  Sparkles,
  Star,
  Sun,
  UserRound,
  X,
} from 'lucide-react'
import movieLensCatalogData from './data/movielens-catalog.json'

type Screen = 'auth' | 'home' | 'recommend' | 'history' | 'compare'
type AuthMode = 'login' | 'register'
type Theme = 'dark' | 'light'

type UserRating = { movieId: number; rating: number; timestamp: number }
type UserProfile = { key: string; displayName: string; email?: string; userId: number | null; ratings: UserRating[] }
type PredictedMovie = { movieId: number; title: string; year: number | null; genres: string[]; predictedRating: number; averageRating: number; ratingCount: number }
type RankingEvaluation = {
  precision_at_10?: number
  recall_at_10?: number
  ndcg_at_10?: number
  item_coverage?: number
  users_evaluated?: number
  negative_candidates_per_user?: number
  relevance_threshold?: number
  candidate_protocol?: string
}
type EvaluationScore = { rmse: number; mae: number; inference_ms_per_1000: number; ranking?: RankingEvaluation }
type EvaluatedModel = { training_seconds: number; sampled_train_rows: number; validation: EvaluationScore; test: EvaluationScore }
type ComparedModel = { name: string; recommendations: PredictedMovie[]; latencyMs: number; metrics: EvaluatedModel }
type ComparisonResponse = { models: Record<string, ComparedModel>; baseline: EvaluatedModel }

type Movie = {
  movieId: number
  title: string
  year: number | null
  averageRating: number
  ratingCount: number
  genres: string[]
  originalGenres: string[]
  poster: string
  hue: string
}

type MovieLensCatalog = {
  statistics: {
    trainRatings: number
    users: number
    movies: number
    averageRating: number
    minimumRating: number
    maximumRating: number
    ratingStep: number
  }
  demoUsers: Array<{ userId: number; ratings: UserRating[] }>
  catalog: Array<Omit<Movie, 'poster' | 'hue' | 'originalGenres'>>
}

const movieLensData = movieLensCatalogData as MovieLensCatalog
const posterPaths: Record<number, string> = {
  260: '/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg',
  296: '/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg',
  318: '/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg',
  356: '/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg',
  2571: '/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg',
  2959: '/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg',
  48780: '/tRNlZbgNCNOpLpbPEz5L8G8A0JN.jpg',
  58559: '/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
  79132: '/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg',
  109487: '/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg',
  115713: '/dmJW8IAKHKxFNiUnoDR7JfsK7Rp.jpg',
  134130: '/3ndAx3weG6KDkJIRMCi5vXX6Dyb.jpg',
  139385: '/oXUWEc5i3wYyFnL1Ycu8ppxxPvs.jpg',
  164179: '/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg',
  176371: '/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg',
}
const poster = (path?: string) => path ? `https://image.tmdb.org/t/p/w500${path}` : ''
const posterHues = ['#477997', '#8993a1', '#526b7d', '#946c4c', '#a86b65', '#54766d']
const genreLabels: Record<string, string> = {
  Action: 'Hành động', Adventure: 'Phiêu lưu', Animation: 'Hoạt hình', Children: 'Gia đình',
  Comedy: 'Hài', Crime: 'Tội phạm', Documentary: 'Tài liệu', Drama: 'Chính kịch',
  Fantasy: 'Kỳ ảo', 'Film-Noir': 'Noir', Horror: 'Kinh dị', IMAX: 'IMAX',
  Musical: 'Nhạc kịch', Mystery: 'Bí ẩn', Romance: 'Lãng mạn', 'Sci-Fi': 'Khoa học viễn tưởng',
  Thriller: 'Giật gân', War: 'Chiến tranh', Western: 'Viễn Tây',
}
const movieCatalog: Movie[] = movieLensData.catalog.map((movie, index) => ({
  ...movie,
  originalGenres: movie.genres,
  genres: movie.genres.map((genre) => genreLabels[genre] ?? genre),
  poster: poster(posterPaths[movie.movieId]),
  hue: posterHues[index % posterHues.length],
}))
const featuredMovies = movieCatalog.slice(0, 6)
const selectableMovies = movieCatalog
const formatRating = (rating: number) => new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(rating)
const formatCount = (count: number) => new Intl.NumberFormat('vi-VN').format(count)
const ratingStorageKey = (key: string) => `movielens-ratings:${key}`
const readSavedRatings = (key: string): UserRating[] | null => {
  try {
    const stored = window.localStorage.getItem(ratingStorageKey(key))
    return stored ? JSON.parse(stored) as UserRating[] : null
  } catch { return null }
}
const persistRatings = (key: string, ratings: UserRating[]) => {
  try { window.localStorage.setItem(ratingStorageKey(key), JSON.stringify(ratings)) } catch { /* local demo storage may be unavailable */ }
}
const movieById = (movieId: number) => movieCatalog.find((movie) => movie.movieId === movieId)
const dateLabel = (timestamp: number) => new Intl.DateTimeFormat('vi-VN', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(timestamp * 1000))

const models = [
  { key: 'linear', label: 'Linear Regression', short: 'Linear Regression', note: 'Ridge tuyến tính, dự đoán rating từ hồ sơ user, phim, thể loại và thời gian.' },
  { key: 'xgboost', label: 'XGBoost', short: 'XGBoost', note: 'Boosted trees, dự đoán rating từ đặc trưng user, phim, thể loại và thời gian.' },
  { key: 'svd', label: 'SVD / Matrix Factorization', short: 'SVD / MF', note: 'Phân rã ma trận user-item với latent factors và bias theo thời gian.' },
  { key: 'ncf', label: 'NCF', short: 'NCF', note: 'Neural Collaborative Filtering với embedding user-item và đặc trưng thời gian.' },
]

function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand ${compact ? 'brand-compact' : ''}`}>
      <span className="brand-mark"><Clapperboard size={22} strokeWidth={2.3} /></span>
      {!compact && <span className="brand-name">MovieLens <b>AI</b></span>}
    </div>
  )
}

function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const nextTheme = theme === 'dark' ? 'light' : 'dark'
  return (
    <button className="theme-toggle" type="button" aria-label={`Chuyển sang giao diện ${nextTheme === 'light' ? 'sáng' : 'tối'}`} title={`Chuyển sang giao diện ${nextTheme === 'light' ? 'sáng' : 'tối'}`} onClick={onToggle}>
      {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
      <span>{theme === 'dark' ? 'Giao diện sáng' : 'Giao diện tối'}</span>
    </button>
  )
}

function Poster({ movie, className = '' }: { movie: Movie; className?: string }) {
  const [failed, setFailed] = useState(false)
  return (
    <div className={`poster ${className}`} style={{ '--poster-hue': movie.hue } as CSSProperties}>
      {movie.poster && !failed ? <img src={movie.poster} alt={`Poster ${movie.title}`} loading="lazy" onError={() => setFailed(true)} /> : <div className="poster-fallback"><Film size={27} /><span>{movie.title}</span></div>}
      <span className="poster-shade" />
    </div>
  )
}

function AuthScreen({ onEnter, onDemoEnter, onLogin, onRegister, theme, onToggleTheme }: { onEnter: () => void; onDemoEnter: (userId: number) => void; onLogin: (identifier: string) => boolean; onRegister: (displayName: string, email: string) => void; theme: Theme; onToggleTheme: () => void }) {
  const [mobileMode, setMobileMode] = useState<AuthMode>('login')
  const [showPassword, setShowPassword] = useState(false)
  const [notice, setNotice] = useState('')

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const values = new FormData(event.currentTarget)
    if (mobileMode === 'login') {
      if (!onLogin(String(values.get('login-email') ?? '').trim())) setNotice('Nhập User ID từ 1 đến 5 hoặc email đã đăng ký trên trình duyệt này.')
      return
    }
    const password = String(values.get('register-password') ?? '')
    if (password !== String(values.get('register-confirm-password') ?? '')) {
      setNotice('Mật khẩu xác nhận chưa khớp.')
      return
    }
    onRegister(String(values.get('register-name') ?? '').trim(), String(values.get('register-email') ?? '').trim())
  }

  return (
    <main className="auth-page">
      <div className="auth-topbar"><Logo /><div className="auth-topbar-actions"><div className="auth-help">Bản xem trước giao diện <span className="help-dot"><Sparkles size={13} /></span></div><ThemeToggle theme={theme} onToggle={onToggleTheme} /></div></div>
      <section className={`auth-shell auth-shell-${mobileMode}`}>
        <div className="auth-art" aria-hidden="true">
          <div className="art-glow" />
          <div className="art-poster art-poster-one"><Poster movie={featuredMovies[0]} /></div>
          <div className="art-poster art-poster-two"><Poster movie={featuredMovies[2]} /></div>
          <div className="art-poster art-poster-three"><Poster movie={featuredMovies[1]} /></div>
          <div className="art-caption"><span className="art-line" /><span>PHIM HAY<br />ĐANG CHỜ BẠN</span></div>
        </div>
        <div className="auth-panel">
          <div className="auth-form-brand"><Logo compact /><span>MovieLens <b>AI</b></span></div>
          <div className="auth-form-wrap">
            <div className="auth-mode-tabs" role="tablist" aria-label="Chọn hình thức tài khoản">
              <button type="button" role="tab" aria-selected={mobileMode === 'login'} className={mobileMode === 'login' ? 'active' : ''} onClick={() => setMobileMode('login')}>Đăng nhập</button>
              <button type="button" role="tab" aria-selected={mobileMode === 'register'} className={mobileMode === 'register' ? 'active' : ''} onClick={() => setMobileMode('register')}>Đăng ký</button>
            </div>
            <div className="auth-intro">
              <div className="auth-icon">{mobileMode === 'login' ? <Clapperboard size={27} /> : <Sparkles size={27} />}</div>
              <h1>{mobileMode === 'login' ? <>Chào mừng <span>trở lại</span></> : 'Bắt đầu khám phá'}</h1>
              <p>{mobileMode === 'login' ? 'Chọn hồ sơ MovieLens mẫu để xem lịch sử thật trong tập train và gợi ý minh họa.' : 'Tạo hồ sơ demo, thêm phim đã xem và chấm điểm để cá nhân hóa gợi ý.'}</p>
            </div>
            {mobileMode === 'login' ? (
              <>
                <form className="auth-form" autoComplete="off" onSubmit={submit}>
                  <label className="field-label" htmlFor="login-email">User ID MovieLens hoặc email</label>
                  <div className="input-wrap"><Mail size={17} /><input id="login-email" name="login-email" type="text" placeholder="1–5 hoặc email đã đăng ký" autoComplete="off" required /></div>
                  <div className="label-row"><label className="field-label" htmlFor="login-password">Mật khẩu</label><button className="text-link" type="button" onClick={() => setNotice('Đăng nhập hiện là bản demo, chưa có xác thực máy chủ.')}>Quên mật khẩu?</button></div>
                  <div className="input-wrap"><LockKeyhole size={17} /><input id="login-password" name="login-password" type={showPassword ? 'text' : 'password'} placeholder="Mật khẩu demo bất kỳ" autoComplete="new-password" required /><button className="input-action" type="button" aria-label={showPassword ? 'Ẩn mật khẩu' : 'Hiện mật khẩu'} onClick={() => setShowPassword((value) => !value)}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></div>
                  <label className="remember-row"><input type="checkbox" defaultChecked /><span className="custom-check"><Check size={11} /></span><span>Ghi nhớ đăng nhập</span></label>
                  <button className="primary-button auth-submit" type="submit">Đăng nhập <ArrowRight size={16} /></button>
                </form>
                <div className="sample-account-block">
                  <div className="sample-account-heading">Hồ sơ MovieLens mẫu <span>User ID 1–5</span></div>
                  <div className="sample-account-grid">{movieLensData.demoUsers.map((user) => <button type="button" key={user.userId} onClick={() => onDemoEnter(user.userId)}><b>{user.userId}</b><span>{formatCount(user.ratings.length)} đánh giá</span></button>)}</div>
                  <p>Lịch sử lấy từ tập train đã xử lý. Chọn hồ sơ để mở trang gợi ý.</p>
                </div>
              </>
            ) : (
              <form className="auth-form" autoComplete="off" onSubmit={submit}>
                <label className="field-label" htmlFor="register-name">Họ và tên</label>
                <div className="input-wrap"><UserRound size={17} /><input id="register-name" name="register-name" type="text" placeholder="Nguyễn Văn A" autoComplete="off" required /></div>
                <label className="field-label" htmlFor="register-email">Email</label>
                <div className="input-wrap"><Mail size={17} /><input id="register-email" name="register-email" type="email" placeholder="ban@email.com" autoComplete="off" required /></div>
                <label className="field-label" htmlFor="register-password">Mật khẩu</label>
                <div className="input-wrap"><LockKeyhole size={17} /><input id="register-password" name="register-password" type={showPassword ? 'text' : 'password'} placeholder="Tối thiểu 8 ký tự" autoComplete="new-password" required minLength={8} /><button className="input-action" type="button" aria-label="Hiện hoặc ẩn mật khẩu" onClick={() => setShowPassword((value) => !value)}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></div>
                <label className="field-label" htmlFor="register-confirm-password">Xác nhận mật khẩu</label>
                <div className="input-wrap"><LockKeyhole size={17} /><input id="register-confirm-password" name="register-confirm-password" type={showPassword ? 'text' : 'password'} placeholder="Nhập lại mật khẩu" autoComplete="new-password" required minLength={8} /></div>
                <label className="terms-row"><input type="checkbox" required /><span>Tôi đồng ý với điều khoản sử dụng demo</span></label>
                <button className="primary-button auth-submit" type="submit">Tạo tài khoản <ArrowRight size={16} /></button>
                <p className="local-account-note">Hồ sơ và điểm đánh giá chỉ lưu trong trình duyệt này; mật khẩu không được gửi lên máy chủ.</p>
              </form>
            )}
            <div className="or-divider"><span />{mobileMode === 'login' ? 'hoặc tiếp tục với' : 'hoặc đăng ký với'}<span /></div>
            <div className="social-row"><button type="button" className="social-button" onClick={onEnter}><GoogleMark />Google</button><button type="button" className="social-button" onClick={onEnter}><Github size={17} />GitHub</button></div>
            <p className="auth-switch">{mobileMode === 'login' ? 'Chưa có tài khoản?' : 'Đã có tài khoản?'} <button className="text-link" onClick={() => setMobileMode(mobileMode === 'login' ? 'register' : 'login')}>{mobileMode === 'login' ? 'Đăng ký ngay' : 'Đăng nhập'}</button></p>
          </div>
        </div>
      </section>
      {notice && <div className="toast"><CheckCircle2 size={17} />{notice}<button onClick={() => setNotice('')} aria-label="Đóng"><X size={15} /></button></div>}
      <button className="demo-entry" onClick={onEnter}>Xem giao diện demo <ArrowRight size={14} /></button>
    </main>
  )
}

function GoogleMark() {
  return <svg className="google-mark" viewBox="0 0 24 24" aria-hidden="true"><path fill="#4285F4" d="M21.6 12.23c0-.71-.06-1.39-.18-2.05H12v3.88h5.38a4.6 4.6 0 0 1-2 3.02v2.53h3.24c1.9-1.75 2.98-4.33 2.98-7.38Z"/><path fill="#34A853" d="M12 22c2.7 0 4.97-.9 6.62-2.39l-3.24-2.53c-.9.6-2.04.96-3.38.96-2.6 0-4.8-1.75-5.59-4.1H3.06v2.6A10 10 0 0 0 12 22Z"/><path fill="#FBBC05" d="M6.41 13.94a6 6 0 0 1 0-3.88v-2.6H3.06a10 10 0 0 0 0 9.08l3.35-2.6Z"/><path fill="#EA4335" d="M12 5.96c1.47 0 2.78.5 3.82 1.5l2.87-2.87A9.57 9.57 0 0 0 12 2a10 10 0 0 0-8.94 5.46l3.35 2.6C7.2 7.71 9.4 5.96 12 5.96Z"/></svg>
}

function Sidebar({ screen, onNavigate, onLogout, profile }: { screen: Screen; onNavigate: (next: Screen) => void; onLogout: () => void; profile: UserProfile }) {
  const initials = profile.displayName.trim().slice(0, 1).toUpperCase() || 'M'
  const profileLabel = profile.userId === null ? 'Tài khoản mới' : `MovieLens User ID ${profile.userId}`
  return (
    <aside className="sidebar">
      <button className="sidebar-brand" onClick={() => onNavigate('home')}><Logo compact /><span className="brand-wordmark">MovieLens <b>AI</b></span></button>
      <div className="nav-caption">KHÔNG GIAN CỦA BẠN</div>
      <nav className="side-nav" aria-label="Điều hướng chính">
        <button className={screen === 'home' ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate('home')}><Home size={18} /><span>Trang chủ</span></button>
        <button className={screen === 'recommend' || screen === 'history' ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate('recommend')}><Sparkles size={18} /><span>Gợi ý phim</span><span className="nav-new">AI</span></button>
        <button className={screen === 'compare' ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate('compare')}><BarChart3 size={18} /><span>So sánh mô hình</span></button>
      </nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-user"><div className="avatar">{initials}</div><div className="user-copy"><strong>{profile.displayName}</strong><span>{profileLabel}</span></div><button className="logout-button" onClick={onLogout} aria-label="Đăng xuất"><ArrowRight size={17} /></button></div>
    </aside>
  )
}

function Topbar({ search, setSearch, onLogout, onMenu, profile, theme, onToggleTheme }: { search: string; setSearch: (value: string) => void; onLogout: () => void; onMenu: () => void; profile: UserProfile; theme: Theme; onToggleTheme: () => void }) {
  const [showNotifications, setShowNotifications] = useState(false)
  const [showProfile, setShowProfile] = useState(false)
  const initials = profile.displayName.trim().slice(0, 1).toUpperCase() || 'M'
  const profileLabel = profile.userId === null ? 'Hồ sơ lưu trên trình duyệt' : `MovieLens User ID ${profile.userId}`
  return (
    <header className="topbar">
      <button className="mobile-menu" aria-label="Mở menu" onClick={onMenu}><Menu size={20} /></button>
      <div className="searchbox"><Search size={17} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Tìm theo tên phim hoặc thể loại..." /></div>
      <div className="topbar-actions"><ThemeToggle theme={theme} onToggle={onToggleTheme} />
        <span className="preview-badge"><i /> GIAO DIỆN DEMO</span>
        <button className={`icon-button notification-button ${showNotifications ? 'selected' : ''}`} aria-label="Thông báo" onClick={() => setShowNotifications((value) => !value)}><Bell size={18} /><i /></button>
        <button className="profile-button" onClick={() => setShowProfile((value) => !value)}><span className="avatar avatar-small">{initials}</span><span className="profile-name">{profile.displayName}</span><ChevronRight size={14} className="profile-chevron" /></button>
        {showNotifications && <div className="popover notification-popover"><strong>Lịch sử gắn với hồ sơ này</strong><p>{profile.ratings.length} lượt đánh giá. Điểm mới sẽ lưu trong localStorage của trình duyệt demo.</p></div>}
        {showProfile && <div className="popover profile-popover"><div className="popover-user"><span className="avatar avatar-small">{initials}</span><span><b>{profile.displayName}</b><small>{profileLabel}</small></span></div><button onClick={onLogout}>Đăng xuất <ArrowRight size={14} /></button></div>}
      </div>
    </header>
  )
}

function MovieCard({ movie, index }: { movie: Movie; index?: number }) {
  return (
    <article className="movie-card">
      <div className="movie-poster-wrap"><Poster movie={movie} /><button className="poster-play" aria-label={`Xem ${movie.title}`}><Film size={16} /></button>{index !== undefined && <span className="rank-badge">{String(index + 1).padStart(2, '0')}</span>}</div>
      <div className="movie-meta"><div className="movie-title-row"><h3>{movie.title}</h3><span className="movie-rating"><Star size={12} fill="currentColor" />{formatRating(movie.averageRating)}</span></div><p>{movie.genres.join(' · ')} <span>{movie.year ? `· ${movie.year}` : ''}</span></p><small className="movie-count">{formatCount(movie.ratingCount)} lượt đánh giá</small></div>
    </article>
  )
}

function HomeScreen({ onRecommend, search }: { onRecommend: () => void; search: string }) {
  const matches = useMemo(() => search.trim() ? movieCatalog.filter((movie) => `${movie.title} ${movie.genres.join(' ')} ${movie.originalGenres.join(' ')}`.toLowerCase().includes(search.toLowerCase())) : featuredMovies, [search])
  const filtered = search.trim() ? matches.slice(0, 24) : matches
  return (
    <div className="home-content page-enter">
      <section className="hero-card">
        <img className="hero-image" src="https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=1800&q=85" alt="Trái Đất nhìn từ không gian" />
        <div className="hero-overlay" />
        <div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" />
        <div className="hero-content"><div className="eyebrow"><Sparkles size={13} /> MOVIELENS 25M · ĐIỂM 0,5–5</div><h1>Mỗi bộ phim hay<br />bắt đầu từ <span>gu của bạn.</span></h1><p>Khám phá danh mục MovieLens 25M. Điểm trung bình trên trang được tính từ dữ liệu train sau khi lọc.</p><button className="primary-button hero-button" onClick={onRecommend}>Tìm phim cho tôi <ArrowRight size={16} /></button></div>
        <div className="hero-index"><span>01</span><i /> MOVIELENS AI</div>
        <div className="hero-dots"><span className="on" /><span /><span /></div>
      </section>
      <section className="section-block featured-section">
        <div className="section-heading"><div><div className="eyebrow section-eyebrow">MOVIELENS 25M · TẬP TRAIN</div><h2>{search.trim() ? `Kết quả cho “${search}”` : 'Phim được đánh giá nhiều nhất'} <span className="heading-count">{String(search.trim() ? matches.length : filtered.length).padStart(2, '0')}</span></h2><p>{search.trim() ? `Tìm thấy ${formatCount(matches.length)} phim; hiển thị tối đa 24 kết quả.` : `Điểm trung bình tính trên thang 0,5–5 từ ${formatCount(movieLensData.statistics.trainRatings)} lượt train.`}</p></div><button className="subtle-link" onClick={onRecommend}>Mở danh mục <ArrowRight size={15} /></button></div>
        {filtered.length ? <div className="featured-grid">{filtered.map((movie) => <MovieCard key={movie.movieId} movie={movie} />)}</div> : <div className="empty-state"><Search size={20} /><span>Không tìm thấy phim phù hợp với “{search}”.</span></div>}
      </section>
    </div>
  )
}

function RecommendationScreen({ search, setSearch, profile, onSaveRating, onRemoveRating, onViewAll }: {
  search: string
  setSearch: (value: string) => void
  profile: UserProfile
  onSaveRating: (movieId: number, rating: number) => void
  onRemoveRating: (movieId: number) => void
  onViewAll: () => void
}) {
  const [editingMovieId, setEditingMovieId] = useState<number | null>(null)
  const [draftRating, setDraftRating] = useState('4')
  const [selectedModel, setSelectedModel] = useState('xgboost')
  const [notice, setNotice] = useState('')
  const [predictions, setPredictions] = useState<PredictedMovie[]>([])
  const [loadingPredictions, setLoadingPredictions] = useState(false)
  const [apiError, setApiError] = useState('')
  const [predictionLatency, setPredictionLatency] = useState(0)
  const [modelMetrics, setModelMetrics] = useState<EvaluatedModel | undefined>(undefined)
  const filtered = useMemo(() => selectableMovies.filter((movie) => `${movie.title} ${movie.genres.join(' ')} ${movie.originalGenres.join(' ')}`.toLowerCase().includes(search.toLowerCase())), [search])
  const ratingsByMovie = useMemo(() => new Map(profile.ratings.map((entry) => [entry.movieId, entry])), [profile.ratings])
  const historyRows = useMemo(() => [...profile.ratings].sort((a, b) => b.timestamp - a.timestamp).slice(0, 8), [profile.ratings])
  const editingMovie = editingMovieId === null ? undefined : movieById(editingMovieId)

  useEffect(() => {
    const controller = new AbortController()
    setLoadingPredictions(true)
    setApiError('')
    fetch('/api/recommendations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({ userId: profile.userId, history: profile.ratings, model: selectedModel }),
    }).then(async (response) => {
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Không thể tải dự đoán.')
      return data as { recommendations: PredictedMovie[]; latencyMs: number }
    }).then((data) => {
      if (!controller.signal.aborted) {
        setPredictions(data.recommendations)
        setPredictionLatency(data.latencyMs)
      }
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setApiError(error instanceof Error ? error.message : 'Không thể kết nối API dự đoán.')
    }).finally(() => {
      if (!controller.signal.aborted) setLoadingPredictions(false)
    })
    return () => controller.abort()
  }, [profile.key, profile.userId, profile.ratings, selectedModel])

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/models', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Model metrics unavailable')
        return response.json() as Promise<{ models: Array<{ key: string; test?: EvaluationScore; validation?: EvaluationScore; trainingSeconds?: number; sampledTrainRows?: number }> }>
      })
      .then((data) => {
        const selected = data.models.find((model) => model.key === selectedModel)
        if (selected?.test && !controller.signal.aborted) {
          setModelMetrics({
            training_seconds: selected.trainingSeconds ?? 0,
            sampled_train_rows: selected.sampledTrainRows ?? 0,
            validation: selected.validation ?? selected.test,
            test: selected.test,
          })
        }
      })
      .catch(() => { if (!controller.signal.aborted) setModelMetrics(undefined) })
    return () => controller.abort()
  }, [selectedModel])

  const startEditing = (movie: Movie) => {
    setEditingMovieId(movie.movieId)
    setDraftRating(String(ratingsByMovie.get(movie.movieId)?.rating ?? 4))
  }
  const saveRating = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!editingMovie) return
    onSaveRating(editingMovie.movieId, Number(draftRating))
    setNotice(`Đã lưu ${editingMovie.title} · ${formatRating(Number(draftRating))}/5. Gợi ý đã được cập nhật theo lịch sử này.`)
    setEditingMovieId(null)
    window.setTimeout(() => setNotice(''), 3000)
  }
  const ratingOptions = Array.from({ length: 10 }, (_, index) => (index + 1) / 2)

  return (
    <div className="recommend-page page-enter">
      <section className="recommend-heading"><div><div className="eyebrow section-eyebrow"><Sparkles size={13} /> MOVIELENS 25M · TẬP TRAIN</div><h1>Gợi ý phim <span>cá nhân hóa</span></h1><p>Thêm phim đã xem, chấm điểm và xem danh sách gợi ý demo thay đổi theo lịch sử của hồ sơ.</p></div><div className="heading-orb"><Clapperboard size={24} /></div></section>
      <section className="dataset-summary" aria-label="Quy mô dữ liệu đã xử lý">
        <div><b>{formatCount(movieLensData.statistics.trainRatings)}</b><span>Lượt đánh giá train</span></div>
        <div><b>{formatCount(movieLensData.statistics.users)}</b><span>User sau lọc</span></div>
        <div><b>{formatCount(movieLensData.statistics.movies)}</b><span>Phim sau lọc</span></div>
        <div><b>{formatRating(movieLensData.statistics.minimumRating)}–{formatRating(movieLensData.statistics.maximumRating)}</b><span>Thang điểm · TB {formatRating(movieLensData.statistics.averageRating)}/5</span></div>
      </section>
      <p className="dataset-method-note">Hồ sơ User ID 1–5 dùng đánh giá trong tập train. Điểm mới của tài khoản demo lưu cục bộ trên trình duyệt và không sửa dataset gốc.</p>
      <div className="recommend-workspace">
        <section className="picker-panel panel-card">
          <div className="panel-heading"><div className="step-number">01</div><div><h2>Phim đã xem và đánh giá</h2><p>Thêm hoặc sửa điểm từ thang 0,5 đến 5 sao.</p></div></div>
          <div className="history-heading"><div><span>LỊCH SỬ CỦA BẠN</span><b>{formatCount(profile.ratings.length)} phim</b></div><small>{profile.userId === null ? 'Lưu trong trình duyệt' : `Hồ sơ MovieLens User ID ${profile.userId}`}</small></div>
          {historyRows.length ? <div className="history-list">{historyRows.map((entry) => {
            const movie = movieById(entry.movieId)
            if (!movie) return null
            return <div className="history-row" key={entry.movieId}><Poster movie={movie} className="history-poster" /><div className="history-copy"><b>{movie.title}</b><span><Star size={11} fill="currentColor" /> {formatRating(entry.rating)} <i>·</i> {dateLabel(entry.timestamp)}</span></div><button className="history-edit" onClick={() => startEditing(movie)}>Sửa</button><button className="history-remove" aria-label={`Xóa đánh giá ${movie.title}`} onClick={() => onRemoveRating(movie.movieId)}><X size={14} /></button></div>
          })}</div> : <div className="history-empty">Bạn chưa thêm phim nào. Tìm phim bên dưới rồi chấm điểm để bắt đầu.</div>}
          {profile.ratings.length > 0 && <button className="history-view-all" type="button" onClick={onViewAll}>Xem toàn bộ danh sách <span>{formatCount(profile.ratings.length)} phim</span><ArrowRight size={13} /></button>}
          {editingMovie && <form className="rating-editor" onSubmit={saveRating}><div className="rating-editor-title"><b>{ratingsByMovie.has(editingMovie.movieId) ? 'Sửa đánh giá' : 'Thêm phim đã xem'}</b><button type="button" aria-label="Đóng" onClick={() => setEditingMovieId(null)}><X size={14} /></button></div><div className="rating-editor-movie"><Poster movie={editingMovie} className="history-poster" /><span>{editingMovie.title}<small>{editingMovie.genres.slice(0, 3).join(' · ') || 'Chưa có thể loại'}</small></span></div><label htmlFor="movie-rating">Đánh giá của bạn</label><div className="rating-editor-controls"><select id="movie-rating" value={draftRating} onChange={(event) => setDraftRating(event.target.value)}>{ratingOptions.map((rating) => <option key={rating} value={rating}>{formatRating(rating)} / 5 sao</option>)}</select><button className="primary-button" type="submit">Lưu đánh giá <Check size={14} /></button></div></form>}
          <div className="picker-search"><Search size={15} /><input aria-label="Tìm phim trong MovieLens" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Tìm trong 14.427 phim..." /></div>
          <div className="movie-list-caption">DANH MỤC · {formatCount(filtered.length)} KẾT QUẢ</div>
          <div className="selectable-list">{filtered.slice(0, 30).map((movie) => {
            const existingRating = ratingsByMovie.get(movie.movieId)
            return <button className={`selectable-movie ${existingRating ? 'is-selected' : ''}`} key={movie.movieId} onClick={() => startEditing(movie)}><Poster movie={movie} className="list-poster" /><span className="selectable-copy"><b>{movie.title}</b><small>{movie.year ?? 'Năm chưa rõ'} <span>·</span> {movie.genres.join(', ')} <span>·</span> {formatCount(movie.ratingCount)} lượt</small></span>{existingRating ? <span className="list-rating"><Star size={11} fill="currentColor" />{formatRating(existingRating.rating)}</span> : <span className="add-rating-label"><Plus size={12} /> Thêm</span>}</button>
          })}</div>
          <div className="picker-note"><span className="note-dot" /> Chọn phim để thêm mới hoặc cập nhật điểm đã chấm.</div>
        </section>

        <div className="results-column">
          <section className="model-panel panel-card">
            <div className="panel-heading panel-heading-compact"><div className="step-number">02</div><div><h2>Mô hình dự đoán rating</h2><p>Chọn mô hình đã huấn luyện để tính Top 10 rating cho hồ sơ hiện tại.</p></div></div>
            <div className="model-options" role="group" aria-label="Chọn mô hình gợi ý">{models.map((model) => <button key={model.key} className={`model-option ${selectedModel === model.key ? 'selected' : ''}`} onClick={() => setSelectedModel(model.key)}><span className="model-option-name">{model.short}</span>{model.key === 'ncf' && <span className="model-tag">AI</span>}</button>)}</div>
            <div className="model-description"><span className="model-dot" />{models.find((model) => model.key === selectedModel)?.note}<span className="demo-label">MODEL ĐÃ HUẤN LUYỆN</span></div>
          </section>
          <section className="results-panel panel-card">
            <div className="results-heading"><div className="results-title"><div className="step-number">03</div><div><div className="results-title-row"><h2>Top 10 phim theo rating dự đoán</h2><span className="result-count">TOP 10</span></div><p>{profile.ratings.length ? `Đầu vào gồm ${formatCount(profile.ratings.length)} rating, thể loại và timestamp.` : 'Chưa có lịch sử; dùng trung bình train làm hồ sơ ban đầu.'}</p></div></div><span className="result-filter">RMSE TEST: {(() => { const value = modelMetrics?.test.rmse; return value === undefined ? '—' : formatRating(value) })()}</span></div>
            <div className="personalization-note"><Sparkles size={13} /> Điểm dưới đây do {models.find((model) => model.key === selectedModel)?.label} dự đoán; phim đã xem được loại khỏi danh sách. Thời gian API hiện tại: {predictionLatency.toFixed(1)} ms.</div>
            {loadingPredictions && <div className="history-empty">Đang tính dự đoán từ model...</div>}
            {apiError && <div className="api-error">{apiError}<span>Khởi động API bằng <code>npm run api</code>.</span></div>}
            {!loadingPredictions && !apiError && predictions.length === 0 && <div className="history-empty">Không có phim phù hợp để gợi ý cho lịch sử này.</div>}
            {!loadingPredictions && !apiError && <div className="recommendation-grid">{predictions.map((prediction, index) => {
              const movie = movieById(prediction.movieId)
              if (!movie) return null
              return <article className="recommendation-card" key={movie.movieId}><div className="rec-poster-wrap"><Poster movie={movie} className="rec-poster" /><span className="rec-rank">{String(index + 1).padStart(2, '0')}</span><button className="rec-add" aria-label={`Thêm ${movie.title} vào phim đã xem`} onClick={() => startEditing(movie)}><Plus size={14} /></button></div><div className="rec-card-copy"><h3>{movie.title}</h3><div className="rec-genres">{movie.genres.slice(0, 2).map((genre) => <span key={genre}>{genre}</span>)}</div><div className="rec-score"><span>Rating dự đoán</span><b>{formatRating(prediction.predictedRating)} <Star size={12} fill="currentColor" /></b></div></div></article>
            })}</div>}
            <div className="results-footer"><span><span className="results-footer-dot" /> Thêm hoặc đổi rating bên trái để gọi lại model với lịch sử mới.</span></div>
          </section>
        </div>
      </div>
      {notice && <div className="toast"><CheckCircle2 size={17} />{notice}<button onClick={() => setNotice('')} aria-label="Đóng"><X size={15} /></button></div>}
    </div>
  )
}

function RatedMoviesScreen({ search, setSearch, profile, onNavigate, onSaveRating, onRemoveRating }: {
  search: string
  setSearch: (value: string) => void
  profile: UserProfile
  onNavigate: (next: Screen) => void
  onSaveRating: (movieId: number, rating: number) => void
  onRemoveRating: (movieId: number) => void
}) {
  const [editingMovieId, setEditingMovieId] = useState<number | null>(null)
  const [draftRating, setDraftRating] = useState('4')
  const [notice, setNotice] = useState('')
  const entries = useMemo(() => [...profile.ratings]
    .sort((left, right) => right.timestamp - left.timestamp)
    .flatMap((entry) => {
      const movie = movieById(entry.movieId)
      return movie ? [{ entry, movie }] : []
    }), [profile.ratings])
  const query = search.trim().toLocaleLowerCase('vi')
  const filtered = entries.filter(({ movie }) => !query || `${movie.title} ${movie.year ?? ''} ${movie.genres.join(' ')} ${movie.originalGenres.join(' ')}`.toLocaleLowerCase('vi').includes(query))
  const ratingOptions = Array.from({ length: 10 }, (_, index) => (index + 1) / 2)

  const saveEdit = (event: FormEvent<HTMLFormElement>, movie: Movie) => {
    event.preventDefault()
    onSaveRating(movie.movieId, Number(draftRating))
    setEditingMovieId(null)
    setNotice(`Đã cập nhật điểm cho ${movie.title}.`)
    window.setTimeout(() => setNotice(''), 3000)
  }

  return (
    <div className="rated-movies-page page-enter">
      <button className="rated-movies-back" type="button" onClick={() => onNavigate('recommend')}><ArrowLeft size={15} /> Quay lại gợi ý phim</button>
      <section className="recommend-heading"><div><div className="eyebrow section-eyebrow"><Star size={13} /> HỒ SƠ {profile.displayName.toLocaleUpperCase('vi')}</div><h1>Phim đã xem <span>và đánh giá</span></h1><p>Toàn bộ phim trong lịch sử của hồ sơ này, sắp xếp theo thời gian đánh giá gần nhất.</p></div><div className="heading-orb"><Film size={24} /></div></section>
      <section className="rated-movies-panel panel-card">
        <div className="rated-movies-toolbar"><div><h2>Danh sách đã đánh giá</h2><p>{formatCount(filtered.length)} / {formatCount(entries.length)} phim</p></div><label className="picker-search rated-movies-search"><Search size={15} /><input aria-label="Tìm trong danh sách đã đánh giá" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Tìm tên phim hoặc thể loại..." /></label></div>
        {filtered.length ? <div className="rated-movies-list">{filtered.map(({ entry, movie }) => (
          <article className="rated-movie-item" key={entry.movieId}>
            <div className="rated-movie-row"><Poster movie={movie} className="rated-movie-poster" /><div className="rated-movie-copy"><b>{movie.title}</b><span>{movie.year ?? 'Năm chưa rõ'} · {movie.genres.join(' · ') || 'Chưa phân loại'}</span><small><Star size={12} fill="currentColor" /> {formatRating(entry.rating)} / 5 <i>·</i> {dateLabel(entry.timestamp)}</small></div><div className="rated-movie-actions"><button className="history-edit" type="button" onClick={() => { setEditingMovieId(editingMovieId === movie.movieId ? null : movie.movieId); setDraftRating(String(entry.rating)) }}>{editingMovieId === movie.movieId ? 'Đóng' : 'Sửa'}</button><button className="history-remove" type="button" aria-label={`Xóa đánh giá ${movie.title}`} title="Xóa đánh giá" onClick={() => { onRemoveRating(movie.movieId); if (editingMovieId === movie.movieId) setEditingMovieId(null) }}><X size={15} /></button></div></div>
            {editingMovieId === movie.movieId && <form className="rating-editor rated-movie-editor" onSubmit={(event) => saveEdit(event, movie)}><div className="rating-editor-title"><b>Sửa đánh giá · {movie.title}</b><button type="button" aria-label="Đóng" onClick={() => setEditingMovieId(null)}><X size={14} /></button></div><label htmlFor={`movie-rating-${movie.movieId}`}>Điểm của bạn</label><div className="rating-editor-controls"><select id={`movie-rating-${movie.movieId}`} value={draftRating} onChange={(event) => setDraftRating(event.target.value)}>{ratingOptions.map((rating) => <option key={rating} value={rating}>{formatRating(rating)} / 5 sao</option>)}</select><button className="primary-button" type="submit">Lưu đánh giá <Check size={14} /></button></div></form>}
          </article>
        ))}</div> : <div className="rated-movies-empty">{entries.length ? 'Không tìm thấy phim khớp với từ khóa.' : 'Hồ sơ này chưa có phim nào được đánh giá.'}</div>}
      </section>
      {notice && <div className="toast"><CheckCircle2 size={17} />{notice}<button onClick={() => setNotice('')} aria-label="Đóng"><X size={15} /></button></div>}
    </div>
  )
}

type ChartSeries = { key: string; label: string; color: string }
type ChartRow = { label: string; values: Record<string, number | undefined> }

function MetricChart({ title, description, series, rows, fixedMax }: { title: string; description: string; series: ChartSeries[]; rows: ChartRow[]; fixedMax?: number }) {
  const scales = Object.fromEntries(series.map((item) => [
    item.key,
    fixedMax ?? Math.max(...rows.map((row) => row.values[item.key] ?? 0), 0.01),
  ]))
  return (
    <section className="metric-chart panel-card">
      <div className="metric-chart-heading"><div><h3>{title}</h3><p>{description}</p></div><BarChart3 size={17} /></div>
      <div className="chart-legend">{series.map((item) => <span key={item.key}><i style={{ '--series-color': item.color } as CSSProperties} />{item.label}</span>)}</div>
      <div className="chart-rows">{rows.map((row) => (
        <div className="chart-row" key={row.label}>
          <strong>{row.label}</strong>
          <div className="chart-row-bars">{series.map((item) => {
            const value = row.values[item.key]
            const width = value === undefined ? 0 : Math.min(100, Math.max(0, value / scales[item.key] * 100))
            return <div className="chart-bar-line" key={item.key}>
              <span className="chart-track"><i style={{ width: `${width}%`, '--series-color': item.color } as CSSProperties} /></span>
              <small>{value === undefined ? '—' : value.toFixed(3)}</small>
            </div>
          })}</div>
        </div>
      ))}</div>
    </section>
  )
}

function ComparisonScreen({ profile }: { profile: UserProfile }) {
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({ userId: profile.userId, history: profile.ratings, model: 'ncf' }),
    }).then(async (response) => {
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Không thể tải kết quả so sánh.')
      return data as ComparisonResponse
    }).then((data) => {
      if (!controller.signal.aborted) setComparison(data)
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Không thể kết nối API.')
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [profile.key, profile.userId, profile.ratings])

  const compareRows = models.map((model) => ({ config: model, result: comparison?.models[model.key] }))
  const fmt = (value?: number) => value === undefined ? '—' : new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(value)
  const seconds = (value?: number) => value === undefined ? '—' : `${value.toFixed(1)} giây`
  const chartRows: ChartRow[] = comparison ? [
    ...models.map((model) => {
      const result = comparison.models[model.key]?.metrics.test
      return { label: model.short, values: { rmse: result?.rmse, mae: result?.mae } }
    }),
    { label: 'Baseline phim', values: { rmse: comparison.baseline.test.rmse, mae: comparison.baseline.test.mae } },
  ] : []
  const rankingRows: ChartRow[] = comparison ? [
    ...models.map((model) => {
      const ranking = comparison.models[model.key]?.metrics.test.ranking
      return { label: model.short, values: { ndcg: ranking?.ndcg_at_10, recall: ranking?.recall_at_10, precision: ranking?.precision_at_10 } }
    }),
    { label: 'Baseline phim', values: {
      ndcg: comparison.baseline.test.ranking?.ndcg_at_10,
      recall: comparison.baseline.test.ranking?.recall_at_10,
      precision: comparison.baseline.test.ranking?.precision_at_10,
    } },
  ] : []
  const runtimeRows: ChartRow[] = comparison ? models.map((model) => {
    const metrics = comparison.models[model.key]?.metrics
    return { label: model.short, values: { training: metrics?.training_seconds, inference: metrics?.test.inference_ms_per_1000 } }
  }) : []
  const rankingMeta = comparison?.baseline.test.ranking ?? compareRows.map(({ result }) => result?.metrics.test.ranking).find(Boolean)
  const hasRanking = rankingRows.some((row) => row.values.ndcg !== undefined)
  const bestValidationRmse = compareRows.map(({ config, result }) => ({
    label: config.short,
    value: result?.metrics.validation.rmse,
    testValue: result?.metrics.test.rmse,
  })).filter((row) => row.value !== undefined).sort((left, right) => (left.value ?? Infinity) - (right.value ?? Infinity))[0]
  const bestValidationNdcg = compareRows.map(({ config, result }) => ({
    label: config.short,
    value: result?.metrics.validation.ranking?.ndcg_at_10,
    testValue: result?.metrics.test.ranking?.ndcg_at_10,
  })).filter((row) => row.value !== undefined).sort((left, right) => (right.value ?? -Infinity) - (left.value ?? -Infinity))[0]
  const baselineNdcg = rankingRows.find((row) => row.label === 'Baseline phim')?.values.ndcg
  const baselineValidationNdcg = comparison?.baseline.validation.ranking?.ndcg_at_10

  return (
    <div className="compare-page page-enter">
      <section className="recommend-heading"><div><div className="eyebrow section-eyebrow"><BarChart3 size={13} /> ĐÁNH GIÁ TRÊN MOVIELENS 25M</div><h1>So sánh <span>mô hình</span></h1><p>Đánh giá dự đoán rating và thứ hạng Top‑10 trên dữ liệu giữ lại theo thời gian; bên dưới là gợi ý trực tiếp cho hồ sơ hiện tại.</p></div><div className="heading-orb"><BarChart3 size={24} /></div></section>
      <div className="compare-context"><Sparkles size={15} /><span>Hồ sơ <b>{profile.displayName}</b> · {formatCount(profile.ratings.length)} đánh giá đầu vào. Các chỉ số tổng hợp dùng cùng tập test MovieLens 25M; danh sách gợi ý bên dưới dùng lịch sử hồ sơ này.</span></div>
      {loading && <div className="history-empty">Đang tải số liệu và tạo danh sách so sánh...</div>}
      {error && <div className="api-error">{error}<span>Khởi động API bằng <code>npm run api</code>.</span></div>}
      {comparison && <>
        <div className="comparison-chart-grid">
          <MetricChart title="Sai số dự đoán rating" description="RMSE và MAE càng thấp, rating ước lượng càng gần điểm thật. Điểm được tính trên test." series={[
            { key: 'rmse', label: 'RMSE ↓', color: '#8178ff' },
            { key: 'mae', label: 'MAE ↓', color: '#36bfd0' },
          ]} rows={chartRows} />
          <MetricChart title="Chất lượng gợi ý Top‑10" description="NDCG, Recall và Precision càng cao càng tốt; trung bình theo user." series={[
            { key: 'ndcg', label: 'NDCG ↑', color: '#8178ff' },
            { key: 'recall', label: 'Recall ↑', color: '#36bfd0' },
            { key: 'precision', label: 'Precision ↑', color: '#e9ad58' },
          ]} rows={rankingRows} fixedMax={1} />
          <MetricChart title="Chi phí tính toán" description="Thời gian thấp hơn giúp triển khai nhẹ hơn; inference đo trên 1.000 rating test." series={[
            { key: 'training', label: 'Huấn luyện (giây) ↓', color: '#8178ff' },
            { key: 'inference', label: 'Inference (ms/1.000) ↓', color: '#36bfd0' },
          ]} rows={runtimeRows} />
        </div>
        <div className="evaluation-takeaways">
          <div><span>Chọn theo sai số rating (validation)</span><b>{bestValidationRmse?.label ?? 'Chưa có số liệu'}</b><small>{bestValidationRmse?.value === undefined ? 'Chưa có RMSE validation.' : `RMSE val ${fmt(bestValidationRmse.value)}; RMSE test ${fmt(bestValidationRmse.testValue)}.`}</small></div>
          <div><span>Top‑10 trong các mô hình cá nhân hóa</span><b>{bestValidationNdcg?.label ?? 'Chưa có số liệu'}</b><small>{bestValidationNdcg?.value === undefined ? 'Chưa có NDCG@10 validation.' : `NDCG val ${fmt(bestValidationNdcg.value)}; test ${fmt(bestValidationNdcg.testValue)}; baseline val/test ${fmt(baselineValidationNdcg)} / ${fmt(baselineNdcg)}.`}</small></div>
          <p>Dùng validation để chọn cấu hình/mô hình; giữ test cho báo cáo cuối. Đây là một lần chia theo thời gian và ứng viên lấy mẫu, vì vậy nên chạy thêm seed hoặc mốc thời gian để kiểm tra độ ổn định.</p>
        </div>
        <div className="ranking-protocol"><Sparkles size={15} /><span><b>Thiết lập Top‑10:</b> rating từ 4 sao là phim liên quan; mỗi user được đánh giá có toàn bộ phim đã rating trong holdout và 100 phim chưa xem lấy mẫu làm ứng viên. Cùng user, ứng viên và ngưỡng cho mọi mô hình. Đây là đánh giá ranking trên ứng viên lấy mẫu, không phải toàn bộ danh mục.</span></div>
        {!hasRanking && <div className="history-empty">Chưa có số liệu Top‑10. Chạy lại bước huấn luyện để tạo Precision@10, Recall@10 và NDCG@10.</div>}
        <section className="compare-metrics panel-card">
          <div className="panel-heading"><div className="step-number"><BarChart3 size={13} /></div><div><h2>Bảng số liệu và chi phí</h2><p>RMSE/MAE dùng dự đoán rating; Precision/Recall/NDCG đánh giá thứ tự phim gợi ý. Hai nhóm trả lời hai câu hỏi khác nhau.</p></div></div>
          <div className="compare-table-wrap"><table className="compare-table"><thead><tr><th>Mô hình</th><th>RMSE test ↓</th><th>MAE test ↓</th><th>NDCG@10 test ↑</th><th>Recall@10 test ↑</th><th>Precision@10 test ↑</th><th>NDCG@10 val</th><th>Recall@10 val</th><th>Precision@10 val</th><th>Coverage</th><th>RMSE val</th><th>MAE val</th><th>Huấn luyện</th><th>Latency / 1.000 rating</th></tr></thead><tbody>
            {compareRows.map(({ config, result }) => {
              const ranking = result?.metrics.test.ranking
              const validationRanking = result?.metrics.validation.ranking
              return <tr key={config.key}><th>{config.label}<small>{result?.metrics.sampled_train_rows ? `${formatCount(result.metrics.sampled_train_rows)} mẫu train` : '—'}</small></th><td>{fmt(result?.metrics.test.rmse)}</td><td>{fmt(result?.metrics.test.mae)}</td><td>{fmt(ranking?.ndcg_at_10)}</td><td>{fmt(ranking?.recall_at_10)}</td><td>{fmt(ranking?.precision_at_10)}</td><td>{fmt(validationRanking?.ndcg_at_10)}</td><td>{fmt(validationRanking?.recall_at_10)}</td><td>{fmt(validationRanking?.precision_at_10)}</td><td>{ranking?.item_coverage === undefined ? '—' : `${fmt(ranking.item_coverage * 100)}%`}</td><td>{fmt(result?.metrics.validation.rmse)}</td><td>{fmt(result?.metrics.validation.mae)}</td><td>{seconds(result?.metrics.training_seconds)}</td><td>{result ? `${fmt(result.metrics.test.inference_ms_per_1000)} ms` : '—'}</td></tr>
            })}
            <tr className="baseline-row"><th>Baseline: trung bình phim<small>Không cá nhân hóa</small></th><td>{fmt(comparison.baseline.test.rmse)}</td><td>{fmt(comparison.baseline.test.mae)}</td><td>{fmt(comparison.baseline.test.ranking?.ndcg_at_10)}</td><td>{fmt(comparison.baseline.test.ranking?.recall_at_10)}</td><td>{fmt(comparison.baseline.test.ranking?.precision_at_10)}</td><td>{fmt(comparison.baseline.validation.ranking?.ndcg_at_10)}</td><td>{fmt(comparison.baseline.validation.ranking?.recall_at_10)}</td><td>{fmt(comparison.baseline.validation.ranking?.precision_at_10)}</td><td>{comparison.baseline.test.ranking?.item_coverage === undefined ? '—' : `${fmt(comparison.baseline.test.ranking.item_coverage * 100)}%`}</td><td>{fmt(comparison.baseline.validation.rmse)}</td><td>{fmt(comparison.baseline.validation.mae)}</td><td>—</td><td>—</td></tr>
          </tbody></table></div>
          {rankingMeta && <p className="table-footnote">Top‑10 dùng {formatCount(rankingMeta.users_evaluated ?? 0)} user ở test, ngưỡng liên quan {rankingMeta.relevance_threshold ?? 4} sao và {rankingMeta.negative_candidates_per_user ?? 100} phim chưa xem lấy mẫu mỗi user.</p>}
        </section>
        <section className="compare-recommendations"><div className="section-heading"><div><div className="eyebrow section-eyebrow">CÙNG MỘT HỒ SƠ ĐẦU VÀO</div><h2>Phim mỗi mô hình đề xuất</h2><p>Điểm cạnh tên phim là rating dự đoán cho lịch sử xem và đánh giá của hồ sơ này.</p></div></div>
          <div className="compare-model-grid">{compareRows.map(({ config, result }) => <article className="compare-model-card panel-card" key={config.key}><div className="compare-model-heading"><div><span className="model-dot" /><h3>{config.label}</h3></div><span>{result ? `${result.latencyMs.toFixed(1)} ms` : '—'}</span></div>{result?.recommendations.map((item, index) => { const movie = movieById(item.movieId); return <div className="compare-movie-row" key={item.movieId}><b className="compare-rank">{index + 1}</b>{movie && <Poster movie={movie} className="compare-poster" />}<span className="compare-movie-title">{item.title}<small>{item.genres.slice(0, 2).join(' · ') || 'Chưa phân loại'}</small></span><strong>{formatRating(item.predictedRating)} <Star size={10} fill="currentColor" /></strong></div>})}</article>)}</div>
        </section>
      </>}
    </div>
  )
}
export default function App() {
  const [screen, setScreen] = useState<Screen>('auth')
  const [search, setSearch] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [theme, setTheme] = useState<Theme>(() => {
    try { return window.localStorage.getItem('movielens-theme') === 'light' ? 'light' : 'dark' } catch { return 'dark' }
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { window.localStorage.setItem('movielens-theme', theme) } catch { /* theme still applies for this session */ }
  }, [theme])

  const navigate = (next: Screen) => {
    setScreen(next)
    setSearch('')
    setSidebarOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
  const activateDemoUser = (userId: number) => {
    const sample = movieLensData.demoUsers.find((user) => user.userId === userId)
    if (!sample) return
    const key = `movielens-user-${userId}`
    setProfile({ key, displayName: `MovieLens User ${userId}`, userId, ratings: readSavedRatings(key) ?? sample.ratings })
    navigate('recommend')
  }
  const localProfiles = (): Array<{ email: string; displayName: string }> => {
    try { return JSON.parse(window.localStorage.getItem('movielens-local-profiles') ?? '[]') as Array<{ email: string; displayName: string }> } catch { return [] }
  }
  const startLocalProfile = (displayName: string, email: string) => {
    const normalizedEmail = email.toLowerCase()
    const key = `local-${normalizedEmail}`
    const profiles = localProfiles().filter((item) => item.email.toLowerCase() !== normalizedEmail)
    profiles.push({ email: normalizedEmail, displayName })
    try { window.localStorage.setItem('movielens-local-profiles', JSON.stringify(profiles)) } catch { /* local demo storage may be unavailable */ }
    setProfile({ key, displayName, email: normalizedEmail, userId: null, ratings: readSavedRatings(key) ?? [] })
    navigate('recommend')
  }
  const login = (identifier: string) => {
    const normalized = identifier.toLowerCase().trim()
    const idMatch = normalized.match(/^(?:movielens\s*)?(?:user(?:\s*id)?\s*)?#?([1-5])$/)
    if (idMatch) { activateDemoUser(Number(idMatch[1])); return true }
    const found = localProfiles().find((item) => item.email.toLowerCase() === normalized)
    if (!found) return false
    const key = `local-${normalized}`
    setProfile({ key, displayName: found.displayName, email: found.email, userId: null, ratings: readSavedRatings(key) ?? [] })
    navigate('recommend')
    return true
  }
  const saveRating = (movieId: number, rating: number) => {
    if (!profile) return
    const next = [...profile.ratings.filter((entry) => entry.movieId !== movieId), { movieId, rating, timestamp: Math.floor(Date.now() / 1000) }]
      .sort((a, b) => a.timestamp - b.timestamp)
    persistRatings(profile.key, next)
    setProfile({ ...profile, ratings: next })
  }
  const removeRating = (movieId: number) => {
    if (!profile) return
    const next = profile.ratings.filter((entry) => entry.movieId !== movieId)
    persistRatings(profile.key, next)
    setProfile({ ...profile, ratings: next })
  }
  const logout = () => {
    setProfile(null)
    navigate('auth')
  }
  const enterGuest = () => {
    setProfile({ key: 'guest', displayName: 'Khách demo', userId: null, ratings: [] })
    navigate('home')
  }

  if (!profile || screen === 'auth') return <AuthScreen onEnter={enterGuest} onDemoEnter={activateDemoUser} onLogin={login} onRegister={startLocalProfile} theme={theme} onToggleTheme={() => setTheme((value) => value === 'dark' ? 'light' : 'dark')} />

  return (
    <div className="app-shell">
      <div className={`sidebar-scrim ${sidebarOpen ? 'visible' : ''}`} onClick={() => setSidebarOpen(false)} />
      <div className={`sidebar-wrap ${sidebarOpen ? 'sidebar-open' : ''}`}><Sidebar screen={screen} onNavigate={navigate} onLogout={logout} profile={profile} /></div>
      <main className="main-area">
        <Topbar search={search} setSearch={setSearch} onLogout={logout} onMenu={() => setSidebarOpen((value) => !value)} profile={profile} theme={theme} onToggleTheme={() => setTheme((value) => value === 'dark' ? 'light' : 'dark')} />
        <div className="page-container">{screen === 'home' ? <HomeScreen onRecommend={() => navigate('recommend')} search={search} /> : screen === 'compare' ? <ComparisonScreen profile={profile} /> : screen === 'history' ? <RatedMoviesScreen search={search} setSearch={setSearch} profile={profile} onNavigate={navigate} onSaveRating={saveRating} onRemoveRating={removeRating} /> : <RecommendationScreen search={search} setSearch={setSearch} profile={profile} onSaveRating={saveRating} onRemoveRating={removeRating} onViewAll={() => navigate('history')} />}</div>
      </main>
    </div>
  )
}
