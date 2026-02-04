# 🔑 Getting DeGirum API Key

**Required for:** Option A (Hybrid) and Option B (Full DeGirum Pipeline)

---

## * Quick Steps

### 1. Sign Up for DeGirum Cloud (FREE)

**Go to:** https://cs.degirum.com

1. Click **"Sign Up"** or **"Get Started"**
2. Fill in your details:
   - Email address
   - Password
   - Organization name (can be your name or company)
3. Verify your email (check inbox/spam)
4. Log in to your account

**Free Tier Includes:**
- [x] 100,000 inferences per month
- [x] Access to all public models
- [x] Cloud inference
- [x] No credit card required

---

### 2. Get Your API Token

Once logged in:

1. **Go to your dashboard** (automatically opens after login)
2. **Find API Settings** or **Account Settings**
3. **Look for "API Token" or "Access Token"**
4. **Click "Generate Token"** or **"Copy Token"**

Your token will look like:
```
dg_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**⚠️ Important:**
- Keep this token secret (don't share publicly)
- You can regenerate it if compromised
- Save it somewhere safe

---

### 3. Configure in Project

**Option 1: Using env.ini file (Recommended)**

Create a file named `env.ini` in your project root:

```bash
# In project directory
cd "Face detection Raspberry"

# Create env.ini file
cat > env.ini << 'EOF'
[DEFAULT]
DEGIRUM_CLOUD_TOKEN = dg_your_actual_token_here
EOF
```

**Or manually create `env.ini`:**

```ini
[DEFAULT]
DEGIRUM_CLOUD_TOKEN = dg_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Replace** `dg_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` with your actual token!

---

**Option 2: Environment Variable**

```bash
# Linux/macOS
export DEGIRUM_CLOUD_TOKEN="dg_your_actual_token_here"

# Windows CMD
set DEGIRUM_CLOUD_TOKEN=dg_your_actual_token_here

# Windows PowerShell
$env:DEGIRUM_CLOUD_TOKEN="dg_your_actual_token_here"
```

---

### 4. Verify Configuration

**Test that your API key works:**

```bash
python -c "
import degirum as dg
import degirum_tools

try:
    token = degirum_tools.get_token()
    print(f'[x] Token found: {token[:10]}...')
    print('[x] Configuration successful!')
except Exception as e:
    print(f'[!] Error: {e}')
    print('Check your env.ini file or environment variable')
"
```

**Expected output:**
```
[x] Token found: dg_xxxxxxx...
[x] Configuration successful!
```

---

### 5. Run Gender Detection Tests

Now you can use Option A and B:

```bash
# Test Option A (Hybrid)
python option_a_hailo_degirum.py --video your_video.mp4

# Test Option B (Full DeGirum)
python option_b_degirum_full.py --video your_video.mp4
```

---

## 🔍 Troubleshooting

### "Authentication failed"

**Check your env.ini file:**
```bash
cat env.ini
```

Should show:
```ini
[DEFAULT]
DEGIRUM_CLOUD_TOKEN = dg_xxxxxxxxx
```

**Common issues:**
- [!] Extra quotes: `"dg_xxx"` (remove quotes)
- [!] Extra spaces: ` dg_xxx ` (remove spaces)
- [!] Wrong section: Use `[DEFAULT]` not `[Other]`
- [!] Typo in key name: Must be exactly `DEGIRUM_CLOUD_TOKEN`

**Correct format:**
```ini
[DEFAULT]
DEGIRUM_CLOUD_TOKEN = dg_your_token_here
```

---

### "Cannot find token"

**Make sure env.ini is in the right place:**
```bash
# Should be in project root
ls -la env.ini

# Output should show the file
-rw-r--r--  1 user  staff  85 Jan  7 10:30 env.ini
```

**If file doesn't exist:**
```bash
# Create it
echo "[DEFAULT]" > env.ini
echo "DEGIRUM_CLOUD_TOKEN = dg_your_token_here" >> env.ini
```

---

### "Token invalid"

1. **Regenerate token** on DeGirum website
2. **Update env.ini** with new token
3. **Verify no extra characters** (spaces, quotes, newlines)
4. **Check token starts with** `dg_`

---

### "No internet connection"

DeGirum cloud requires internet. If you need offline:
- Use **Option Caffe** instead: `python test_gender_caffe.py video.mp4`
- Or set up local DeGirum AI Server (advanced, see DeGirum docs)

---

## 📊 Usage Limits

**Free Tier (No Credit Card):**
- 100,000 inferences/month
- Public models only
- Cloud inference only

**What counts as 1 inference?**
- 1 frame processed = 1 inference
- At 20 FPS for 1 minute = 1,200 inferences
- 100,000 inferences ≈ 83 minutes of video at 20 FPS

**For more usage:**
- Check DeGirum pricing: https://degirum.ai/pricing
- Consider local inference (requires DeGirum AI Server license)

---

## 🔐 Security Best Practices

[x] **DO:**
- Keep token in `env.ini` (add to .gitignore)
- Regenerate if accidentally shared
- Use environment variables in production
- Keep token secret

[!] **DON'T:**
- Commit `env.ini` to git
- Share token publicly
- Hardcode token in scripts
- Email or message token

**Add to .gitignore:**
```bash
echo "env.ini" >> .gitignore
```

---

## * Alternative: Local Inference

If you don't want cloud dependency:

**Option 1: Use Offline Models**
```bash
# Use Caffe (no API key needed)
python test_gender_caffe.py video.mp4

# Use InsightFace (no API key needed)
pip install insightface onnxruntime
python test_gender_insightface.py video.mp4
```

**Option 2: DeGirum AI Server (Advanced)**
- Requires license purchase
- Runs models locally on your hardware
- No internet needed after setup
- See: https://degirum.com/products/ai-server

---

## 📝 Quick Reference

### env.ini Template:
```ini
[DEFAULT]
DEGIRUM_CLOUD_TOKEN = dg_paste_your_token_here
```

### Test Commands:
```bash
# Verify token
python -c "import degirum_tools; print(degirum_tools.get_token())"

# Test Option A
python option_a_hailo_degirum.py --video test.mp4

# Test Option B  
python option_b_degirum_full.py --video test.mp4
```

### Links:
- Sign up: https://cs.degirum.com
- Pricing: https://degirum.ai/pricing
- Documentation: https://degirum.com/docs

---

## [x] Checklist

Setup complete when you can check all:
- [ ] Signed up for DeGirum account
- [ ] Received and copied API token
- [ ] Created `env.ini` file with token
- [ ] Token verification works
- [ ] Option A or B script runs successfully

---

**Need Help?**
- DeGirum support: https://degirum.com/support
- Check TEAM_TESTING_GUIDE.md for troubleshooting
- Verify `env.ini` format carefully

**Ready to test? Run:**
```bash
python option_a_hailo_degirum.py --video your_video.mp4
```
