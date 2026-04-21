def check_network_status(region: str):
    return {
        "status": "ok",
        "region": region,
        "latency_ms": 12,
        "packet_loss_pct": 0.1,
    }


def get_customer_info(phone: str):
    return {
        "phone": phone,
        "name": "Sayin Musteri",
        "plan": "Fiber 100 Mbps",
        "status": "active",
        "balance_try": 0,
        "invoice_due": "2026-05-01",
    }


# Tüm soru-cevap verilerinden türetilmiş kapsamlı bilgi tabanı
_KB = [
    {
        "tags": ["internet yavaş", "bağlantı yavaş", "hız testi", "modem restart"],
        "title": "Yavaş İnternet Giderme Rehberi",
        "content": (
            "Modeminizi 30 saniye kapatıp yeniden başlatın, kablo bağlantılarını kontrol edin "
            "ve hız testi yapın. Sorun devam ederse 444 0 375 hattından destek alabilirsiniz."
        ),
    },
    {
        "tags": ["fatura", "itiraz", "hatalı fatura"],
        "title": "Fatura İtirazı",
        "content": (
            "Online İşlemler üzerinden 'Fatura İtirazı' başlığından dilekçenizi iletebilirsiniz. "
            "İtiraz başvurusu sonrası 15 iş günü içinde dönüş sağlanır."
        ),
    },
    {
        "tags": ["fatura", "ödeme", "ödenmemiş", "borç"],
        "title": "Ödenmemiş Fatura Ödeme",
        "content": (
            "Ödenmemiş faturanızı mobil uygulama > Faturalarım menüsünden kredi kartı, "
            "banka kartı veya havale ile ödeyebilirsiniz. Son ödeme tarihini geçtiyse gecikme faizi uygulanabilir."
        ),
    },
    {
        "tags": ["fatura", "yüksek", "detay"],
        "title": "Yüksek Fatura Sebebi",
        "content": (
            "Yüksek gelen faturanın detayını Online İşlemler > Fatura Detayı sayfasından madde madde "
            "inceleyebilirsiniz. Ek paket, roaming veya kullanım aşımı kalemleri nedeniyle olabilir. "
            "Hatalı gördüğünüz kalem için Fatura İtirazı başlatabilirsiniz."
        ),
    },
    {
        "tags": ["fatura", "son ödeme", "tarih"],
        "title": "Fatura Son Ödeme Tarihi",
        "content": (
            "Son ödeme tarihi faturanın üzerinde ve mobil uygulamada 'Faturalarım' kısmında yazılıdır. "
            "Otomatik ödeme talimatı vererek gecikme yaşamazsınız."
        ),
    },
    {
        "tags": ["tarife", "paket değiştir", "tarife değiştir", "tarife yükselt"],
        "title": "Tarife Değişikliği",
        "content": (
            "turktelekom.com.tr veya mobil uygulama üzerinden 'Tarifem' menüsünden mevcut tarifenizi "
            "değiştirebilirsiniz. Taahhütlü hatlarda cayma bedeli uygulanabilir."
        ),
    },
    {
        "tags": ["taahhüt", "süre", "cayma", "erken bozma"],
        "title": "Taahhüt ve Cayma Bilgisi",
        "content": (
            "Taahhütlü hattınızı süresinden önce sonlandırırsanız cayma bedeli tahakkuk eder. "
            "Güncel cayma tutarını mobil uygulamada 'Hesabım > Taahhüt Bilgisi' altında görebilirsiniz."
        ),
    },
    {
        "tags": ["5g", "4.5g", "kapsama", "altyapı"],
        "title": "5G/4.5G Kapsama Sorgulama",
        "content": (
            "Türk Telekom kapsama haritası üzerinden adresinizi girerek 4.5G/5G hizmetlerinin "
            "sağlanıp sağlanmadığını öğrenebilirsiniz."
        ),
    },
    {
        "tags": ["fiber", "altyapı", "adres", "uygunluk"],
        "title": "Fiber Altyapı Uygunluğu",
        "content": (
            "turktelekom.com.tr/altyapi-sorgulama sayfasından adresinizi girerek fiber hizmet "
            "uygunluğunu kontrol edebilirsiniz."
        ),
    },
    {
        "tags": ["roaming", "yurt dışı", "yurtdışı", "paket"],
        "title": "Yurt Dışı Roaming Aktivasyonu",
        "content": (
            "Online İşlemler > Roaming bölümünden gideceğiniz ülkeye uygun paketi aktive edebilirsiniz. "
            "Yolculuk öncesi 24 saat açtırmanız önerilir."
        ),
    },
    {
        "tags": ["modem", "arıza", "değişim", "teknik"],
        "title": "Modem Arızası ve Değişim",
        "content": (
            "Arızalı modem bildiriminizi 444 0 375 üzerinden açın. Teknik ekip uzaktan test sonrası "
            "arızayı doğrularsa değişim ücretsiz yapılır."
        ),
    },
    {
        "tags": ["numara", "taşıma", "porting", "operatör değiş"],
        "title": "Numara Taşıma (MNP)",
        "content": (
            "Bayilere kimliğinizle başvurduktan sonra 1-3 iş günü içinde numara taşıma tamamlanır. "
            "Eski operatördeki borçların kapatılmış olması gerekir."
        ),
    },
    {
        "tags": ["statik ip", "ip adresi", "ek hizmet"],
        "title": "Statik IP Hizmeti",
        "content": (
            "Bireysel müşteriler için statik IP hizmeti aylık ek ücretle sunulur. "
            "Online İşlemler üzerinden 'Ek Hizmetler' bölümünden aktive edebilirsiniz."
        ),
    },
    {
        "tags": ["ek paket", "internet paketi", "satın al", "kota", "kota aşımı"],
        "title": "Ek Paket Satın Alma",
        "content": (
            "Mobil uygulama veya 'Online İşlemler > Ek Paketler' menüsünden mevcut tarifenize uygun "
            "internet, dakika veya SMS paketi satın alabilirsiniz. "
            "Kotanız dolduğunda ek paket aktivasyonuyla hız anında normale döner."
        ),
    },
    {
        "tags": ["dondur", "dondurma", "askıya", "abonelik askıya"],
        "title": "Hat Dondurma / Abonelik Askıya Alma",
        "content": (
            "Askıya alma işlemi için bayiye kimlikle başvuru gereklidir. "
            "En fazla 6 ay boyunca hat dondurulabilir; bu süre içinde sabit ücret alınmaz, "
            "hat numaranız korunur. İşlem aynı kanaldan geri alınabilir."
        ),
    },
    {
        "tags": ["wifi", "şifre", "kablosuz", "wpa"],
        "title": "WiFi Şifre Değişikliği",
        "content": (
            "Modem arayüzüne bilgisayarınızdan 192.168.1.1 adresinden giriş yapıp 'Kablosuz Ayarlar' "
            "bölümünden WPA2 şifresini değiştirebilirsiniz. Varsayılan kullanıcı adı/şifre modemin "
            "altındaki etikettedir."
        ),
    },
    {
        "tags": ["kesinti", "ödedim", "açılmıyor", "reset"],
        "title": "Ödeme Sonrası Hat Açılmıyor",
        "content": (
            "Ödeme sonrası hattın yeniden açılması 5-30 dakika sürebilir. Eğer hâlâ bağlantı yoksa "
            "modemi resetleyin; sorun devam ederse 444 0 375'ten bölgesel arıza kaydı açılabilir."
        ),
    },
]


def query_knowledge_base(query: str, top_k: int = 3):
    q = query.lower()
    scored = []
    for doc in _KB:
        score = sum(1 for tag in doc["tags"] if tag in q)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [d for _, d in scored if _ > 0] or [d for _, d in scored[:top_k]]
    return {"query": query, "results": results[:top_k]}
